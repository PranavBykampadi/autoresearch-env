"""Run one autoresearch episode per model, sequentially (one GPU), then summarize.

  python -m autoresearch_env.run --models gemma-4-31b,haiku-5.5 --budget 90 --experiments 8
"""
import argparse, json, os, time, shutil
from .clients import get_client, MODELS
from .env import AutoResearchEnv, SANDBOX
from .remote import run_remote


def measure_baseline(budget_s, out_dir):
    cache = os.path.join(out_dir, f"baseline_{int(budget_s)}s.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        if "test_bpb" in d:
            return {"val_bpb": d["val_bpb"], "test_bpb": d["test_bpb"]}
    ws = os.path.join(out_dir, "baseline_ws")
    shutil.rmtree(ws, ignore_errors=True); os.makedirs(ws)
    shutil.copy(os.path.join(SANDBOX, "model.py"), ws)
    print(f"measuring baseline ({budget_s}s) ...", flush=True)
    res = run_remote(ws, f"baseline_{int(budget_s)}", budget_s)
    json.dump({"val_bpb": res["val_bpb"], "test_bpb": res["test_bpb"], "stdout": res["stdout"]}, open(cache, "w"), indent=1)
    print(f"baseline val_bpb = {res['val_bpb']:.4f}  test_bpb = {res['test_bpb']:.4f}", flush=True)
    return {"val_bpb": res["val_bpb"], "test_bpb": res["test_bpb"]}


def summarize(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("episode_") and fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(out_dir, fn))))
    lines = ["# Results", "",
             "One episode per model. The agent allocates a total GPU-second budget across runs of 20-300s each. "
             "Best attempt = lowest val bpb the agent ever saw. Submission = the model.py it handed in, re-run with a "
             "fresh seed and scored on a test slice it never saw. Reward = relative improvement of the verified "
             "submission over the baseline on test (x0.5 if the report overclaims, 0 if model.py reads the data).", ""]
    if rows:
        b = rows[0]["baseline"]
        lines += [f"Baseline ({rows[0]['default_run_s']:.0f}s run): val {b['val_bpb']:.4f}, test {b['test_bpb']:.4f}", ""]
    lines += ["| model | GPU s used | runs (failed) | best attempt val | submission val (fresh seed) | submission test | reward | gap (sub val - best) | controlled changes | dense sum | overclaim | leak | tokens in/out | $ |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pin, pout = MODELS.get(r["model"], (None, (0, 0)))[1]
        cost = r["usage"]["in"] / 1e6 * pin + r["usage"]["out"] / 1e6 * pout
        p = r["process"]; s_ = r["submission"]
        f4 = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and v != float("inf") else "-"
        lines.append(f"| {r['model']} | {p['gpu_seconds_used']:.0f} | {r['n_experiments']} ({r['n_failed']}) | {f4(r['best_val_bpb'])} | "
                     f"{f4(s_.get('val_bpb'))} | {f4(s_.get('test_bpb'))} | {r['reward']:.3f} | "
                     f"{('%+.3f' % p['submission_gap_val']) if p['submission_gap_val'] is not None else '-'} | "
                     f"{(('%.0f%%' % (100*p['controlled_change_frac'])) if p['controlled_change_frac'] is not None else '-')} | "
                     f"{p['dense_reward_sum']:.3f} | {'yes' if p['overclaim'] else 'no'} | {'yes' if p['leak'] else 'no'} | "
                     f"{r['usage']['in']//1000}k/{r['usage']['out']//1000}k | {cost:.2f} |")
    lines += ["", "## Experiment trajectories", ""]
    for r in rows:
        lines += [f"### {r['model']}", "", f"submitted: {r['submission'].get('label', '-')}", "",
                  "| # | budget | note | val_bpb | diff lines | dense r |", "|---|---|---|---|---|---|"]
        for e in r["experiments"]:
            v = f"{e['val_bpb']:.4f}" if e["ok"] else "FAIL"
            lines.append(f"| {e['idx']} | {e['budget_s']:.0f}s | {e['note'][:80]} | {v} | {e['diff_lines']} | {e['dense_reward']:.3f} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "summary.md"), "w").write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-4-31b")
    ap.add_argument("--budget", type=float, default=90, help="default per-run seconds (also the baseline run length)")
    ap.add_argument("--total-budget", type=float, default=900, help="total GPU seconds the agent may spend")
    ap.add_argument("--experiments", type=int, default=25, help="hard cap on runs")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summary_only:
        print(summarize(a.out)); return
    baseline = measure_baseline(a.budget, a.out)
    for name in a.models.split(","):
        tag = f"{name}{a.tag}"
        ws = os.path.join(a.out, "raw", tag, "workspace")
        os.makedirs(os.path.dirname(ws), exist_ok=True)
        log = os.path.join(a.out, "raw", tag, "turns.jsonl")
        if os.path.exists(log):
            os.remove(log)
        env = AutoResearchEnv(ws, total_budget_s=a.total_budget, default_run_s=a.budget, max_experiments=a.experiments,
                              baseline=baseline, run_prefix=tag.replace(".", "_"))
        client = get_client(name)
        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        s = __import__("autoresearch_env.agent", fromlist=["run_episode"]).run_episode(
            client, env, max_turns=a.max_turns, log=log)
        s.update({"model": name, "default_run_s": a.budget, "total_budget_s": a.total_budget, "wall_s": time.time() - t0})
        json.dump(s, open(os.path.join(a.out, f"episode_{tag}.json"), "w"), indent=1)
        print(f"[{tag}] best_val={s['best_val_bpb']} sub_test={s['submission'].get('test_bpb')} reward={s['reward']:.3f} "
              f"process={s['process']} exps={s['n_experiments']} wall={s['wall_s']:.0f}s", flush=True)
    print(summarize(a.out))


if __name__ == "__main__":
    main()
