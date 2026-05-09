#!/usr/bin/env bash
# Wait until all 9 training runs finish (each writes done.flag), then
# launch evaluation + summary automatically.
#
# Usage:
#   nohup bash scripts/auto_eval_when_done.sh > outputs/exp_mc_autoeval.log 2>&1 &

set -euo pipefail

REPO_ROOT="/jizhicfs/wuyanning/cluster_data_selection"
cd "${REPO_ROOT}"

PYTHON_BIN="/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python"
N_EXPECTED="${N_EXPECTED:-9}"
POLL_SEC="${POLL_SEC:-60}"

OUT_ROOT="$(cat outputs/latest_exp_mc_out_root.txt)"
echo "[auto-eval] watching OUT_ROOT=${OUT_ROOT}"

while true; do
  n_done=$(find "${OUT_ROOT}/runs" -mindepth 2 -maxdepth 2 -name done.flag 2>/dev/null | wc -l)
  ts=$(date '+%F %T')
  echo "[${ts}] training done ${n_done}/${N_EXPECTED}"
  if [[ "${n_done}" -ge "${N_EXPECTED}" ]]; then
    echo "[auto-eval] all training runs done; starting evaluation."
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[auto-eval] launching evaluation..."
OUT_ROOT="${OUT_ROOT}" bash scripts/eval_exp_mc_all.sh \
  > "${OUT_ROOT}/eval_launch.log" 2>&1 || echo "[auto-eval] eval rc!=0 but continuing"

echo "[auto-eval] summarizing..."
"${PYTHON_BIN}" scripts/summarize_exp_mc.py \
  > "${OUT_ROOT}/summary_stdout.log" 2>&1 || true

echo "[auto-eval] done. summary at ${OUT_ROOT}/summary.md"
echo "[auto-eval] last 80 lines of summary:"
tail -80 "${OUT_ROOT}/summary.md" 2>/dev/null || echo "(no summary.md)"
