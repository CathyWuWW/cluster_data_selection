"""
Lightweight watcher for 1.5B pilot / main runs.
Reads `outputs/latest_exp_mc_1p5b_out_root.txt`, prints step + latest
per-domain dev loss for every run.
"""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/jizhicfs/wuyanning/cluster_data_selection")
PTR = REPO_ROOT / "outputs" / "latest_exp_mc_1p5b_out_root.txt"

STEP_RE = re.compile(r"step=(\d+)/(\d+)\s+loss=([\d\.eE+-]+)")
EVAL_RE = re.compile(
    r"\[step=(\d+)\] eval:.*?math=([\d\.]+).*?code=([\d\.]+).*?weighted=([\d\.]+)"
)


def scan(log: Path):
    if not log.exists():
        return 0, None, None, None, None
    last_step = 0
    last_eval_step = None
    m_loss = c_loss = w_loss = None
    try:
        with log.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    last_step = max(last_step, int(m.group(1)))
                e = EVAL_RE.search(line)
                if e:
                    last_eval_step = int(e.group(1))
                    m_loss = float(e.group(2))
                    c_loss = float(e.group(3))
                    w_loss = float(e.group(4))
    except Exception:
        pass
    return last_step, last_eval_step, m_loss, c_loss, w_loss


def main():
    if not PTR.exists():
        print(f"[err] {PTR} missing")
        sys.exit(1)
    out_root = Path(PTR.read_text().strip())
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    once = "--once" in sys.argv
    while True:
        runs = sorted((out_root / "runs").glob("*"))
        n_done = sum(1 for r in runs if (r / "done.flag").exists())
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] {n_done}/{len(runs)} done  OUT_ROOT={out_root.name}")
        print(f"  {'run':<34} {'step':>6} {'eval@':>6} {'math':>6} {'code':>6} {'w_mc':>6} done")
        print(f"  {'-'*34} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} ----")
        for r in runs:
            log = r / "train.log"
            step, eval_step, m, c, w = scan(log)
            done = "yes" if (r / "done.flag").exists() else " . "
            ms = f"{m:.4f}" if m is not None else "  -   "
            cs = f"{c:.4f}" if c is not None else "  -   "
            ws = f"{w:.4f}" if w is not None else "  -   "
            es = f"{eval_step}" if eval_step is not None else "  -"
            print(f"  {r.name:<34} {step:>6} {es:>6} {ms:>6} {cs:>6} {ws:>6}  {done}")
        if once or (n_done == len(runs) and len(runs) > 0):
            if n_done == len(runs) and len(runs) > 0:
                print("[watch] all done; exit.")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
