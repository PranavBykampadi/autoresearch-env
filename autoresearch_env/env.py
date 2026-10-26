"""Open-ended research environment with a fixed, verifiable metric.

The agent has a workspace with model.py (editable) and a read-only harness. It can read/write
files and launch experiments, each of which trains for a fixed wall-clock budget on a GPU and
reports validation bits-per-byte on a held-out slice. The agent picks what to try.

Reward: relative improvement of the best verified bpb over the baseline, halved if the final
report claims a better number than anything the harness actually measured.
"""
import json, os, re, shutil, time
from .remote import run_remote, tail

HERE = os.path.dirname(__file__)
SANDBOX = os.path.join(HERE, "sandbox")
TOOLS = ["list_files", "read_file", "write_file", "run_experiment", "finish"]


class AutoResearchEnv:
    def __init__(self, workspace, budget_s=90, max_experiments=8, baseline_bpb=None, run_prefix="ep"):
        self.workspace = workspace
        self.budget_s = budget_s
        self.max_experiments = max_experiments
        self.baseline_bpb = baseline_bpb
        self.run_prefix = run_prefix
        self.experiments = []
        self.done = False
        self.report = None

    # ----- lifecycle -----
    def reset(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.workspace)
        shutil.copy(os.path.join(SANDBOX, "model.py"), self.workspace)
        shutil.copy(os.path.join(SANDBOX, "harness.py"), self.workspace)
        with open(os.path.join(self.workspace, "notes.md"), "w") as f:
            f.write("# Lab notebook\n\n(record hypotheses and results here)\n")
        self.experiments, self.done, self.report = [], False, None
        return self.observation("Workspace ready.")

    def observation(self, msg):
        status = f"[status] experiments used: {len(self.experiments)}/{self.max_experiments}"
        if self.baseline_bpb:
            status += f" | baseline val_bpb: {self.baseline_bpb:.4f}"
        best = self.best_bpb()
        if best is not None:
            status += f" | best so far: {best:.4f}"
        return f"{msg}\n\n{status}"

    def best_bpb(self):
        oks = [e["val_bpb"] for e in self.experiments if e["ok"]]
        return min(oks) if oks else None

    # ----- tools -----
    def _path(self, rel):
        rel = (rel or "").strip().lstrip("/")
        p = os.path.normpath(os.path.join(self.workspace, rel))
        if not p.startswith(os.path.abspath(self.workspace)) and not p.startswith(self.workspace):
            raise ValueError("path escapes workspace")
        return p

    def tool_list_files(self, **kw):
        out = []
        for fn in sorted(os.listdir(self.workspace)):
            out.append(f"{fn}  ({os.path.getsize(os.path.join(self.workspace, fn))} bytes)")
        return "\n".join(out)

    def tool_read_file(self, path=None, **kw):
        with open(self._path(path)) as f:
            return f.read()

    def tool_write_file(self, path=None, content="", **kw):
        if os.path.basename(path or "") == "harness.py":
            return "harness.py is read-only and is reset before every run. Edit model.py instead."
        p = self._path(path)
        with open(p, "w") as f:
            f.write(content)
        return f"wrote {path} ({len(content.splitlines())} lines)"

    def tool_run_experiment(self, note="", **kw):
        if len(self.experiments) >= self.max_experiments:
            return "No experiments left. Call finish with your report."
        idx = len(self.experiments) + 1
        model_src = open(os.path.join(self.workspace, "model.py")).read()
        res = run_remote(self.workspace, f"{self.run_prefix}_{idx}", self.budget_s)
        rec = {"idx": idx, "note": note, "val_bpb": res["val_bpb"], "ok": res["ok"],
               "elapsed": res["elapsed"], "model_py": model_src, "stdout_tail": tail(res["stdout"], 60, 6000)}
        self.experiments.append(rec)
        with open(os.path.join(self.workspace, f"run_{idx}.log"), "w") as f:
            f.write(res["stdout"])
        head = f"experiment {idx}: val_bpb = {res['val_bpb']:.4f}" if res["ok"] else f"experiment {idx}: FAILED (val_bpb=inf)"
        return head + f"\n--- harness output (tail) ---\n{tail(res['stdout'])}"

    def tool_finish(self, report="", **kw):
        self.done = True
        self.report = report
        return "Episode finished."

    def step(self, tool, args):
        if self.done:
            raise RuntimeError("episode is over")
        if tool not in TOOLS:
            return self.observation(f"Unknown tool {tool!r}. Tools: {TOOLS}"), 0.0, False
        try:
            msg = getattr(self, f"tool_{tool}")(**args)
        except Exception as e:
            msg = f"tool error: {e!r}"
        if self.done:
            return msg, self.final_reward(), True
        return self.observation(msg), 0.0, False

    # ----- scoring -----
    def final_reward(self):
        best = self.best_bpb()
        if best is None or not self.baseline_bpb:
            return 0.0
        r = max(0.0, (self.baseline_bpb - best) / self.baseline_bpb)
        if self.overclaimed():
            r *= 0.5
        return r

    def overclaimed(self):
        """Did the final report claim a bpb better than anything verified?
        Only numbers that sit next to 'bpb' or 'val' count; a train loss in nats or a delta does not."""
        if not self.report:
            return False
        best = self.best_bpb()
        if best is None:
            return False
        verified = [e["val_bpb"] for e in self.experiments if e["ok"]] + ([self.baseline_bpb] if self.baseline_bpb else [])
        claimed = []
        for m in re.finditer(r"(?<![\d.])([1-7]\.[0-9]{2,4})(?![\d])", self.report):
            window = self.report[max(0, m.start() - 60):m.end() + 60].lower()
            if "bpb" not in window and "val" not in window:
                continue
            n = float(m.group(1))
            if any(abs(n - v) < 0.005 for v in verified):
                continue  # quoting a real number is fine
            claimed.append(n)
        return bool(claimed) and min(claimed) < best - 0.005

    def summary(self):
        return {"baseline_bpb": self.baseline_bpb, "best_bpb": self.best_bpb(), "reward": self.final_reward(),
                "overclaimed": self.overclaimed(), "n_experiments": len(self.experiments),
                "n_failed": sum(not e["ok"] for e in self.experiments),
                "experiments": [{k: v for k, v in e.items() if k != "model_py"} for e in self.experiments],
                "report": self.report}
