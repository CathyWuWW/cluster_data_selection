"""
Watch the Exp-mc launcher progress.

Reads `outputs/latest_exp_mc_out_root.txt` to find the OUT_ROOT, then
prints a per-job status table:
  config         seed   gpu   step    last_eval_weighted   done?
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/jizhicfs/wuyanning/cluster_data_selection")
PTR = REPO_ROOT / "outputs" / "latest_exp_mc_out_root.txt"

STEP_RE = re.compile(r"step=(\d+)/(\d+)\s+loss=([\d\.eE+-]+)")
EVAL_RE = re.compile(
    r"\[step=(\d+)\] eval:.*?math=([\d\.]+).*?code=([\d\.]+).*?weighted=([\d\.]+)"
)


def latest_step_and_eval(log_path: Path) -> tuple[int, float | None, float | None, float | None]:
    if not log_path.exists():
        return 0, None, None, None
    last_step = 0
    last_eval = (None, None, None)  # (math, code, weighted)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    last_step = max(last_step, int(m.group(1)))
                e = EVAL_RE.search(line)
                if e:
                    last_eval = (float(e.group(2)), float(e.group(3)), float(e.group(4)))
    except Exception:
        pass
    return last_step, last_eval[0], last_eval[1], last_eval[2]


def main() -> None:
    if not PTR.exists():
        print(f"[err] no pointer: {PTR}")
        sys.exit(1)
    out_root = Path(PTR.read_text().strip())
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    print(f"[watch] OUT_ROOT={out_root}")

    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        print(f"[err] no runs dir: {runs_dir}")
        sys.exit(1)

    once = "--once" in sys.argv
    while True:
        run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
        rows = []
        n_done = 0
        for rd in run_dirs:
            log = rd / "train.log"
            done = (rd / "done.flag").exists()
            n_done += int(done)
            step, m_loss, c_loss, w_loss = latest_step_and_eval(log)
            rows.append((rd.name, step, m_loss, c_loss, w_loss, done))

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}]  {n_done}/{len(rows)} done")
        print(f"  {'run':<32} {'step':>5}  {'math':>6}  {'code':>6}  {'weighted':>8}  done")
        print(f"  {'-'*32} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  ----")
        for name, step, m, c, w, done in rows:
            ms = f"{m:.4f}" if m is not None else "  -   "
            cs = f"{c:.4f}" if c is not None else "  -   "
            ws = f"{w:.4f}" if w is not None else "   -    "
            ds = "yes" if done else " . "
            print(f"  {name:<32} {step:>5}  {ms:>6}  {cs:>6}  {ws:>8}   {ds}")

        if once:
            return
        if n_done == len(rows) and len(rows) > 0:
            print(f"\n[watch] all {n_done} runs done. exiting.")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
