"""Open-ended research environment with a fixed, verifiable metric. v2.

The agent has a workspace with model.py (editable) and a read-only harness. It can read/write files
and launch experiments. Each experiment trains for a wall-clock budget the agent chooses (20-300s) out
of a total GPU-second allowance for the episode, then reports validation bits-per-byte. The agent
decides what to try, how many runs to make, and how long each one gets.

Scoring (MLGym-style best attempt vs best submission, plus a verification gap check):
  * The agent ends by submitting one experiment's model.py (default: the current file).
  * The environment re-runs that file with a FRESH seed and scores it on a TEST slice the agent never saw.
  * outcome  = relative improvement of the verified test bpb over the baseline test bpb  (the scalar reward)
  * dense    = per-experiment improvement of val bpb over the running best (turn-level credit, logged)
  * process  = verifiable checks, logged and applied as multipliers, never judged by an LLM:
        - overclaim: the report quotes a bpb better than anything the harness printed  -> x0.5
        - leak: model.py touches the data file / val slice                             -> reward 0
        - submission gap: best attempt (val) minus verified submission (val), shows whether
          the agent kept track of its own best result
        - controlled changes: fraction of experiments whose model.py diff was <= 25 lines
"""
import difflib, json, os, re, shutil
from .remote import run_remote, tail

HERE = os.path.dirname(__file__)
SANDBOX = os.path.join(HERE, "sandbox")
TOOLS = ["list_files", "read_file", "write_file", "run_experiment", "finish"]
MIN_RUN, MAX_RUN = 20.0, 300.0
LEAK_PATTERNS = [r"input\.txt", r"\bdata/", r"\bopen\(", r"np\.fromfile", r"frombuffer\(open", r"harness", r"VAL_BYTES", r"TEST_BYTES", r"load_split"]


class AutoResearchEnv:
    def __init__(self, workspace, total_budget_s=900, default_run_s=90, max_experiments=25,
                 baseline=None, run_prefix="ep"):
        self.workspace = workspace
        self.total_budget_s = float(total_budget_s)
        self.default_run_s = float(default_run_s)
        self.max_experiments = max_experiments
        self.baseline = baseline or {}  # {"val_bpb":..., "test_bpb":...}
        self.run_prefix = run_prefix
        self.experiments = []
        self.done = False
        self.report = None
        self.submission = None  # dict with verified numbers
        self.leak = False

    # ----- lifecycle -----
    def reset(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.workspace)
        shutil.copy(os.path.join(SANDBOX, "model.py"), self.workspace)
        shutil.copy(os.path.join(SANDBOX, "harness.py"), self.workspace)
        with open(os.path.join(self.workspace, "notes.md"), "w") as f:
            f.write("# Lab notebook\n\n(record hypotheses and results here)\n")
        self.experiments, self.done, self.report, self.submission, self.leak = [], False, None, None, False
        return self.observation("Workspace ready.")

    def used_s(self):
        return sum(e["budget_s"] for e in self.experiments)

    def remaining_s(self):
        return self.total_budget_s - self.used_s()

    def observation(self, msg):
        status = (f"[status] experiments: {len(self.experiments)} | GPU seconds used: {self.used_s():.0f}/{self.total_budget_s:.0f}"
                  f" (remaining {self.remaining_s():.0f})")
        if self.baseline.get("val_bpb"):
            status += f" | baseline val_bpb: {self.baseline['val_bpb']:.4f}"
        best = self.best_val()
        if best is not None:
            status += f" | best val so far: {best:.4f}"
        return f"{msg}\n\n{status}"

    def best_val(self):
        oks = [e["val_bpb"] for e in self.experiments if e["ok"]]
        return min(oks) if oks else None

    # ----- tools -----
    def _path(self, rel):
        rel = (rel or "").strip().lstrip("/")
        p = os.path.normpath(os.path.join(self.workspace, rel))
        if not p.startswith(os.path.normpath(self.workspace)):
            raise ValueError("path escapes workspace")
        return p

    def tool_list_files(self, **kw):
        return "\n".join(f"{fn}  ({os.path.getsize(os.path.join(self.workspace, fn))} bytes)"
                         for fn in sorted(os.listdir(self.workspace)))

    def tool_read_file(self, path=None, **kw):
        with open(self._path(path)) as f:
            return f.read()

    def tool_write_file(self, path=None, content="", **kw):
        if os.path.basename(path or "") == "harness.py":
            return "harness.py is read-only and is reset before every run. Edit model.py instead."
        with open(self._path(path), "w") as f:
            f.write(content)
        return f"wrote {path} ({len(content.splitlines())} lines)"

    def _leak_check(self, src):
        return [p for p in LEAK_PATTERNS if re.search(p, src)]

    def tool_run_experiment(self, note="", budget=None, **kw):
        if len(self.experiments) >= self.max_experiments:
            return "Experiment limit reached. Call finish."
        try:
            b = float(budget) if budget not in (None, "") else self.default_run_s
        except ValueError:
            return f"budget must be a number of seconds between {MIN_RUN:.0f} and {MAX_RUN:.0f}"
        b = max(MIN_RUN, min(MAX_RUN, b))
        if b > self.remaining_s() + 1e-6:
            if self.remaining_s() < MIN_RUN:
                return f"Only {self.remaining_s():.0f} GPU seconds left, below the {MIN_RUN:.0f}s minimum. Call finish."
            return f"Only {self.remaining_s():.0f} GPU seconds left; pick a budget <= that."
        model_src = open(os.path.join(self.workspace, "model.py")).read()
        hits = self._leak_check(model_src)
        if hits:
            self.leak = True
            return ("model.py appears to read the data file or the harness internals (" + ", ".join(hits) +
                    "). That is against the rules; this run was not executed and the episode is flagged. "
                    "Use only the train_bytes passed to train().")
        idx = len(self.experiments) + 1
        res = run_remote(self.workspace, f"{self.run_prefix}_{idx}", b)
        prev_best = self.best_val()
        prev_src = self.experiments[-1]["model_py"] if self.experiments else open(os.path.join(SANDBOX, "model.py")).read()
        diff_lines = sum(1 for l in difflib.unified_diff(prev_src.splitlines(), model_src.splitlines(), lineterm="", n=0)
                         if l[:1] in "+-" and not l.startswith(("+++", "---")))
        dense = 0.0
        if res["ok"] and self.baseline.get("val_bpb"):
            ref = prev_best if prev_best is not None else self.baseline["val_bpb"]
            dense = max(0.0, (ref - res["val_bpb"]) / self.baseline["val_bpb"])
        rec = {"idx": idx, "note": note, "budget_s": b, "val_bpb": res["val_bpb"], "test_bpb": res["test_bpb"],
               "ok": res["ok"], "elapsed": res["elapsed"], "diff_lines": diff_lines, "dense_reward": dense,
               "model_py": model_src, "stdout_tail": tail(res["stdout"], 60, 6000)}
        self.experiments.append(rec)
        with open(os.path.join(self.workspace, f"run_{idx}.log"), "w") as f:
            f.write(res["stdout"])
        head = (f"experiment {idx} ({b:.0f}s): val_bpb = {res['val_bpb']:.4f}" if res["ok"]
                else f"experiment {idx} ({b:.0f}s): FAILED (val_bpb=inf)")
        return head + f"\n--- harness output (tail) ---\n{tail(res['stdout'])}"

    def tool_finish(self, report="", submit=None, **kw):
        self.done = True
        self.report = report
        # which model.py is the submission?
        src, label = open(os.path.join(self.workspace, "model.py")).read(), "current model.py"
        if submit not in (None, ""):
            try:
                k = int(submit)
                match = [e for e in self.experiments if e["idx"] == k]
                if match:
                    src, label = match[0]["model_py"], f"experiment {k}"
            except ValueError:
                pass
        self.submission = {"label": label, "model_py": src, "verified": False}
        if self.leak or self._leak_check(src):
            self.leak = True
            return "Episode finished. Submission not verified: leak flag set."
        # verification: fresh seed, the submission's budget (or the default), scored on test
        budget = self.default_run_s
        for e in self.experiments:
            if e["model_py"] == src:
                budget = e["budget_s"]
        vws = self.workspace + "_verify"
        shutil.rmtree(vws, ignore_errors=True); os.makedirs(vws)
        with open(os.path.join(vws, "model.py"), "w") as f:
            f.write(src)
        res, err = None, None
        for _ in range(2):
            try:
                res = run_remote(vws, f"{self.run_prefix}_verify", budget, seed=1234)
                break
            except Exception as e:
                err = repr(e)
        if res is None:
            self.submission.update({"verified": False, "error": err})
            return "Episode finished. Verification run failed: " + str(err)
        self.submission.update({"verified": True, "budget_s": budget, "val_bpb": res["val_bpb"],
                                "test_bpb": res["test_bpb"], "ok": res["ok"], "stdout_tail": tail(res["stdout"], 40, 3000)})
        return "Episode finished. Submission re-run with a fresh seed for scoring."

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
    def outcome(self):
        s = self.submission
        bt = self.baseline.get("test_bpb")
        if not s or not s.get("verified") or not s.get("ok") or not bt:
            return 0.0
        return max(0.0, (bt - s["test_bpb"]) / bt)

    def final_reward(self):
        if self.leak:
            return 0.0
        r = self.outcome()
        if self.overclaimed():
            r *= 0.5
        return r

    def overclaimed(self):
        """Does the report claim a validation bpb better than anything the harness measured?

        A number counts as a claim only if the surrounding text talks about val/bpb and does not label it as
        something else (the agent's own holdout, train loss, a delta). A claim is fine if it matches a verified
        number at the precision it was written (2.74 for 2.7484), or the mean of two or three verified runs.
        Everything else that beats the best verified run by more than 0.005 is an overclaim."""
        if not self.report:
            return False
        best = self.best_val()
        if best is None:
            return False
        verified = [e["val_bpb"] for e in self.experiments if e["ok"]] + [v for v in self.baseline.values() if v]
        from itertools import combinations
        derived = [sum(c) / len(c) for k in (2, 3) for c in combinations(verified, k)]
        allowed = verified + derived

        def matches(n, text):
            d = len(text.split(".")[1]) if "." in text else 0
            tol = 0.5 * 10 ** (-d) + 1e-9
            return any(abs(n - v) <= tol for v in allowed)

        claimed = []
        for m in re.finditer(r"(?<![\d.])([1-7]\.[0-9]{2,4})(?![\d])", self.report):
            window = self.report[max(0, m.start() - 60):m.end() + 60].lower()
            if "bpb" not in window and "val" not in window:
                continue
            if any(k in window for k in ("holdout", "hold-out", "train loss", "train_loss", "monitor", "internal", "proxy", "delta", "gap")):
                continue
            n = float(m.group(1))
            if matches(n, m.group(1)):
                continue
            claimed.append(n)
        return bool(claimed) and min(claimed) < best - 0.005

    def process_metrics(self):
        oks = [e for e in self.experiments if e["ok"]]
        best_val = self.best_val()
        sub = self.submission or {}
        gap = (sub["val_bpb"] - best_val) if (sub.get("ok") and best_val is not None) else None
        return {
            "overclaim": self.overclaimed(), "leak": self.leak,
            "submission_gap_val": gap,  # verified submission val minus best attempt val (0 = handed in its best; noise ~0.01)
            "controlled_change_frac": (sum(e["diff_lines"] <= 25 for e in self.experiments) / len(self.experiments)) if self.experiments else None,
            "dense_reward_sum": sum(e["dense_reward"] for e in self.experiments),
            "gpu_seconds_used": self.used_s(),
            "improvement_per_100s": ((self.baseline.get("val_bpb", 0) - best_val) / self.baseline["val_bpb"] / max(self.used_s(), 1) * 100) if (best_val is not None and self.baseline.get("val_bpb")) else None,
        }

    def summary(self):
        return {"baseline": self.baseline, "best_val_bpb": self.best_val(),
                "submission": {k: v for k, v in (self.submission or {}).items() if k != "model_py"},
                "outcome": self.outcome(), "reward": self.final_reward(), "process": self.process_metrics(),
                "n_experiments": len(self.experiments), "n_failed": sum(not e["ok"] for e in self.experiments),
                "experiments": [{k: v for k, v in e.items() if k != "model_py"} for e in self.experiments],
                "report": self.report}
