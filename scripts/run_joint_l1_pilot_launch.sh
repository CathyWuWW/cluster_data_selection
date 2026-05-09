#!/usr/bin/env bash
# Joint Loss Level 1 — pilot launcher
# 4 runs in parallel (one GPU each), all on Qwen2.5-1.5B, m=24, t=1000, seed=42:
#   J0: enabled=true,  lam=0.0,  w=uniform   (sanity: should match B1 baseline)
#   J1: enabled=true,  lam=0.05, w=uniform   (weak pull)
#   J2: enabled=true,  lam=0.2,  w=uniform   (standard pull)
#   J3: enabled=true,  lam=0.2,  w=utility   (utility-weighted pull)
#
# Reuses the meta-cluster artifacts of the existing 1.5B pilot
# (exp_mc_1p5b_pilot_m24_t1000_s42_*) — does NOT recluster.
#
# Usage:
#   nohup bash scripts/run_joint_l1_pilot_launch.sh > outputs/joint_l1_pilot_launch.log 2>&1 & disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"

# --- shared inputs ---
PILOT_OUT="${PILOT_OUT:-outputs/exp_mc_1p5b_pilot_m24_t1000_s42_20260506_162512}"
META_DIR="${PILOT_OUT}/meta_clusters/utility_m24"
PRECOMPUTED_IDS="${META_DIR}/meta_cluster_ids.npy"
BASE_TO_META_CSV="${META_DIR}/base_cluster_to_meta_cluster.csv"
UTILITY_CSV="${UTILITY_CSV:-outputs/utility_mc_v1/utility_matrix_mc_sum_delta.csv}"
N_META_CLUSTERS=24
SEED="${SEED:-42}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}"
TOTAL_ITERS="${TOTAL_ITERS:-1000}"

if [[ ! -f "${PRECOMPUTED_IDS}" ]]; then
  echo "[error] missing ${PRECOMPUTED_IDS}; check PILOT_OUT" >&2; exit 1
fi
if [[ ! -f "${BASE_TO_META_CSV}" ]]; then
  echo "[error] missing ${BASE_TO_META_CSV}" >&2; exit 1
fi
if [[ ! -f "${UTILITY_CSV}" ]]; then
  echo "[error] missing ${UTILITY_CSV}" >&2; exit 1
fi

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME_BASE="joint_l1_pilot_m24_t${TOTAL_ITERS}_s${SEED}_${TAG}"
OUT_ROOT="outputs/${EXP_NAME_BASE}"
GPUS=(0 1 2 3)

mkdir -p "${OUT_ROOT}/runs" "${OUT_ROOT}/logs"
# Symlink meta_clusters for traceability (don't copy, save space)
ln -sfn "$(readlink -f "${PILOT_OUT}/meta_clusters")" "${OUT_ROOT}/meta_clusters"
printf '%s\n' "${OUT_ROOT}" > outputs/latest_joint_l1_out_root.txt

cat > "${OUT_ROOT}/launch_manifest.json" <<EOF
{
  "tag": "${TAG}",
  "model_path": "${MODEL_PATH}",
  "pilot_out": "${PILOT_OUT}",
  "precomputed_ids": "${PRECOMPUTED_IDS}",
  "base_to_meta_csv": "${BASE_TO_META_CSV}",
  "utility_csv": "${UTILITY_CSV}",
  "n_meta_clusters": ${N_META_CLUSTERS},
  "total_iters": ${TOTAL_ITERS},
  "seed": ${SEED},
  "configs": {
    "J0_lam0":           {"lam": 0.0,  "weight_mode": "uniform"},
    "J1_lam0p05":        {"lam": 0.05, "weight_mode": "uniform"},
    "J2_lam0p2":         {"lam": 0.2,  "weight_mode": "uniform"},
    "J3_lam0p2_util":    {"lam": 0.2,  "weight_mode": "utility"}
  },
  "gpus": "${GPUS[*]}"
}
EOF

# --- run launcher ---
declare -a JOBS=(
  "J0_lam0|0.0|uniform"
  "J1_lam0p05|0.05|uniform"
  "J2_lam0p2|0.2|uniform"
  "J3_lam0p2_util|0.2|utility"
)

run_job() {
  local idx="$1" entry="$2" gpu="$3"
  IFS='|' read -r run_name lam wmode <<< "${entry}"
  local out_dir="${OUT_ROOT}/runs/${run_name}_s${SEED}"
  local log="${OUT_ROOT}/logs/job_${idx}_${run_name}_s${SEED}_gpu${gpu}.log"

  if [[ -e "${out_dir}/done.flag" ]]; then
    echo "[skip] ${out_dir} already done"; return
  fi
  mkdir -p "${out_dir}"

  echo "[start] job=${idx} ${run_name} lam=${lam} weight=${wmode} gpu=${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    RUN_NAME="${run_name}" \
    PRECOMPUTED_IDS="${PRECOMPUTED_IDS}" \
    OUT_DIR="${out_dir}" \
    SEED="${SEED}" \
    MODEL_PATH="${MODEL_PATH}" \
    TOTAL_ITERS="${TOTAL_ITERS}" \
    JOINT_ENABLED=true \
    JOINT_LAM="${lam}" \
    JOINT_WEIGHT_MODE="${wmode}" \
    JOINT_WEIGHT_CSV="${UTILITY_CSV}" \
    JOINT_BASE_TO_META="${BASE_TO_META_CSV}" \
    MASTER_PORT="$((29741 + idx))" \
    bash scripts/run_joint_l1_pilot_one.sh \
      && touch "${out_dir}/done.flag"
  ) >"${log}" 2>&1
  echo "[end  ] job=${idx} ${run_name} (rc=$?)"
}

idx=0
for entry in "${JOBS[@]}"; do
  gpu="${GPUS[$idx]}"
  run_job "${idx}" "${entry}" "${gpu}" &
  idx=$((idx + 1))
  sleep 3
done
wait

echo "[done] all ${#JOBS[@]} runs finished. OUT_ROOT=${OUT_ROOT}"

# --- post: auto-stage final/ for platform eval ---
if [[ "${AUTO_STAGE:-1}" == "1" ]]; then
  echo "[stage] staging HF ckpts for platform eval..."
  for entry in "${JOBS[@]}"; do
    IFS='|' read -r run_name _lam _wmode <<< "${entry}"
    src="${OUT_ROOT}/runs/${run_name}_s${SEED}/final"
    if [[ -d "${src}" ]]; then
      SRC_HF="${src}" \
      DST_ROOT="${REPO_ROOT}/outputs/platform_eval_ready" \
      EXP_NAME="${EXP_NAME_BASE}__${run_name}_s${SEED}" \
      STEP="${TOTAL_ITERS}" \
      COPY_MODE=copy \
      bash scripts/stage_for_platform_eval.sh \
        || echo "[stage] WARN: staging ${run_name} failed"
    fi
  done
fi
