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
        return json.load(open(cache))["val_bpb"]
    ws = os.path.join(out_dir, "baseline_ws")
    shutil.rmtree(ws, ignore_errors=True); os.makedirs(ws)
    shutil.copy(os.path.join(SANDBOX, "model.py"), ws)
    print(f"measuring baseline ({budget_s}s) ...", flush=True)
    res = run_remote(ws, f"baseline_{int(budget_s)}", budget_s)
    json.dump({"val_bpb": res["val_bpb"], "stdout": res["stdout"]}, open(cache, "w"), indent=1)
    print(f"baseline val_bpb = {res['val_bpb']:.4f}", flush=True)
    return res["val_bpb"]


def summarize(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("episode_") and fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(out_dir, fn))))
    lines = ["# Results", "",
             "One episode per model. Baseline = the starter model.py run under the same budget. "
             "Reward = relative improvement in val bits-per-byte (halved if the report overclaims).", "",
             "| model | budget | baseline bpb | best bpb | improvement | reward | experiments (failed) | overclaim | tokens in/out | $ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pin, pout = MODELS.get(r["model"], (None, (0, 0)))[1]
        cost = r["usage"]["in"] / 1e6 * pin + r["usage"]["out"] / 1e6 * pout
        best = f"{r['best_bpb']:.4f}" if r["best_bpb"] is not None else "-"
        imp = f"{(r['baseline_bpb']-r['best_bpb'])/r['baseline_bpb']:.1%}" if r["best_bpb"] is not None else "-"
        lines.append(f"| {r['model']} | {r['budget_s']:.0f}s | {r['baseline_bpb']:.4f} | {best} | {imp} | {r['reward']:.3f} | "
                     f"{r['n_experiments']} ({r['n_failed']}) | {'yes' if r['overclaimed'] else 'no'} | "
                     f"{r['usage']['in']//1000}k/{r['usage']['out']//1000}k | {cost:.2f} |")
    lines += ["", "## Experiment trajectories", ""]
    for r in rows:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| # | note | val_bpb |")
        lines.append("|---|---|---|")
        for e in r["experiments"]:
            v = f"{e['val_bpb']:.4f}" if e["ok"] else "FAIL"
            lines.append(f"| {e['idx']} | {e['note'][:90]} | {v} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out_dir, "summary.md"), "w").write(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-4-31b")
    ap.add_argument("--budget", type=float, default=90)
    ap.add_argument("--experiments", type=int, default=8)
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
        env = AutoResearchEnv(ws, budget_s=a.budget, max_experiments=a.experiments, baseline_bpb=baseline,
                              run_prefix=tag.replace(".", "_"))
        client = get_client(name)
        print(f"=== {tag} ===", flush=True)
        t0 = time.time()
        s = __import__("autoresearch_env.agent", fromlist=["run_episode"]).run_episode(
            client, env, max_turns=a.max_turns, log=log)
        s.update({"model": name, "budget_s": a.budget, "wall_s": time.time() - t0})
        json.dump(s, open(os.path.join(a.out, f"episode_{tag}.json"), "w"), indent=1)
        print(f"[{tag}] best={s['best_bpb']} reward={s['reward']:.3f} overclaim={s['overclaimed']} "
              f"exps={s['n_experiments']} wall={s['wall_s']:.0f}s", flush=True)
    print(summarize(a.out))


if __name__ == "__main__":
    main()
