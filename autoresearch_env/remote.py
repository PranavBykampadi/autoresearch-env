"""Run the harness on the lab GPU over ssh. Files travel as a tar stream; no rsync on the box."""
import io, os, re, subprocess, tarfile, time

HOST = os.environ.get("AR_SSH_HOST", "mind-gpu01")
REMOTE_ROOT = os.environ.get("AR_REMOTE_ROOT", "~/autoresearch")
GPU = os.environ.get("AR_GPU", "1")
HERE = os.path.dirname(__file__)
HARNESS = os.path.join(HERE, "sandbox", "harness.py")


def _tar_bytes(workspace):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for fn in os.listdir(workspace):
            if fn.startswith(".") or fn == "data":
                continue
            t.add(os.path.join(workspace, fn), arcname=fn)
        t.add(HARNESS, arcname="harness.py")  # always the canonical harness, overwrites any edit
    return buf.getvalue()


def run_remote(workspace, run_id, budget_s, seed=0):
    """Returns dict(stdout, val_bpb, elapsed, ok)."""
    rdir = f"{REMOTE_ROOT}/runs/{run_id}"
    script = (f"mkdir -p {rdir} && tar xz -C {rdir} && cd {rdir} && ln -sfn ../../data data && "
              f"source {REMOTE_ROOT}/.venv/bin/activate && "
              f"CUDA_VISIBLE_DEVICES={GPU} timeout {int(budget_s) + 150} python -I harness.py --budget {budget_s} --seed {seed} 2>&1")
    t0 = time.time()
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, script], input=_tar_bytes(workspace),
                       capture_output=True, timeout=budget_s + 240)
    out = p.stdout.decode(errors="replace")
    m = re.search(r"FINAL_VAL_BPB=([0-9.]+|inf)", out)
    bpb = float(m.group(1)) if m else float("inf")
    mt = re.search(r"FINAL_TEST_BPB=([0-9.]+|inf)", out)
    tbpb = float(mt.group(1)) if mt else float("inf")
    # the agent never sees the test number
    agent_out = re.sub(r"FINAL_TEST_BPB=.*\n?", "", out)
    return {"stdout": agent_out, "val_bpb": bpb, "test_bpb": tbpb, "elapsed": time.time() - t0,
            "ok": bpb != float("inf"), "rc": p.returncode}


def tail(text, n_lines=40, max_chars=3000):
    lines = text.strip().splitlines()
    s = "\n".join(lines[-n_lines:])
    return s[-max_chars:]
