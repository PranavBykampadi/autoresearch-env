"""Agent loop. Actions are tagged blocks in plain text so the same protocol works for models
with and without native tool calling:

<action tool="write_file" path="model.py">
...file content...
</action>

<action tool="run_experiment" note="try 4 layers"></action>
"""
import re, json, time

SYSTEM = """You are an ML researcher running experiments on a GPU. Your goal: get the lowest possible
validation bits-per-byte (val_bpb) for a byte-level language model on a small text corpus, within a
fixed training time budget per experiment. Everything in model.py is yours to change: architecture,
optimizer, learning rate schedule, batch size, context length, precision, torch.compile, data sampling.
The harness (harness.py) is read-only: it loads data, calls build_model() and train(), and evaluates.

Tools (use exactly one per message, as a tagged block):
<action tool="list_files"></action>
<action tool="read_file" path="model.py"></action>
<action tool="write_file" path="model.py">FULL FILE CONTENT</action>
<action tool="run_experiment" note="one line on what you're testing"></action>
<action tool="finish">REPORT: what you tried, what worked, best val_bpb, and what you'd try next</action>

Rules:
- You have a limited number of experiments. Each one trains for the budget and reports val_bpb.
- Keep the build_model(vocab_size) and train(model, train_bytes, budget_s, device) entry points.
- train() must respect budget_s (wall clock). Runs that go far over are scored as failures.
- Write the whole file when you use write_file; there is no patch tool.
- Keep notes.md updated with hypotheses and results so your report is grounded.
- Be a scientist: change one or two things at a time, read the harness output, and build on what works.
- Only report numbers the harness actually printed."""

ACTION_RE = re.compile(r'<action\s+tool="(\w+)"([^>]*)>(.*?)</action>', re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_action(text):
    """Return (tool, args) for the last action block, or None."""
    ms = list(ACTION_RE.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    tool = m.group(1)
    args = dict(ATTR_RE.findall(m.group(2)))
    body = m.group(3)
    if tool == "write_file":
        # strip a single leading newline and an optional code fence
        body = body.lstrip("\n")
        fence = re.match(r"```[a-zA-Z]*\n(.*?)```\s*$", body, re.S)
        if fence:
            body = fence.group(1)
        args["content"] = body
    elif tool == "finish":
        args["report"] = body.strip()
    return tool, args


def compact_history(messages, keep_last=2):
    """Replace old write_file bodies with a stub so the transcript fits in a 32k context."""
    writes = [i for i, m in enumerate(messages) if m["role"] == "assistant" and 'tool="write_file"' in m["content"]]
    for i in writes[:-keep_last]:
        messages[i]["content"] = ACTION_RE.sub(
            lambda mm: mm.group(0) if mm.group(1) != "write_file" else
            f'<action tool="write_file"{mm.group(2)}>[file content omitted from history; use read_file to see the current version]</action>',
            messages[i]["content"])
    return messages


def run_episode(client, env, max_turns=40, verbose=True, log=None):
    obs = env.reset()
    intro = (f"{obs}\n\nFiles: model.py (baseline, editable), harness.py (read-only), notes.md.\n"
             f"Per-experiment training budget: {env.budget_s:.0f}s on one NVIDIA L40S. "
             f"Experiments available: {env.max_experiments}.\n"
             f"Start by reading model.py and harness.py, then plan your experiments.")
    messages = [{"role": "user", "content": intro}]
    usage = {"in": 0, "out": 0}
    turns = []
    reward, done = 0.0, False
    for t in range(max_turns):
        text, u = client.chat(messages, system=SYSTEM)
        usage["in"] += u["in"]; usage["out"] += u["out"]
        messages.append({"role": "assistant", "content": text})
        act = parse_action(text)
        if act is None:
            fb = "No <action> block found. Reply with exactly one action block."
            tool, args = None, {}
        else:
            tool, args = act
            fb, reward, done = env.step(tool, args)
        turns.append({"turn": t + 1, "tool": tool, "args": {k: (v if k not in ("content",) else f"<{len(v)} chars>") for k, v in args.items()},
                      "response": text, "feedback": fb})
        if verbose:
            tag = f"{tool}({args.get('path') or args.get('note','')[:60]})" if tool else "no-action"
            best = env.best_bpb()
            print(f"  turn {t+1}: {tag}" + (f" | best={best:.4f}" if best is not None else ""), flush=True)
        if log:
            with open(log, "a") as f:
                f.write(json.dumps(turns[-1]) + "\n")
        if done:
            break
        messages.append({"role": "user", "content": fb})
        messages = compact_history(messages)
    if not done:
        # out of turns: score whatever was verified
        env.tool_finish(report="(agent ran out of turns)")
        reward = env.final_reward()
    s = env.summary()
    s.update({"usage": usage, "turns": len(turns), "transcript": turns})
    return s
