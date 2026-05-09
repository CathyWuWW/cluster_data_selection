#!/usr/bin/env bash
# Launcher for 1.5B pretraining-scale Exp-mc runs.
#
# Two presets via env STAGE:
#   STAGE=pilot : 1000 step x max_len 1024 x ga 4  ~= 4M tokens per run
#                (purpose: verify 1.5B fits on H20, PMP wiring OK, ckpt format)
#   STAGE=main  : 12000 step x max_len 1024 x ga 4 ~= 49M tokens per run
#                (purpose: real comparison, still single-seed first)
#
# Runs 3 configs in parallel (one per GPU):
#   - hidden_base
#   - hidden_utility_meta  (m=24 over math+code utility)
#   - hidden_random_meta   (m=24 random regroup)
#
# Usage:
#   STAGE=pilot nohup bash scripts/run_exp_mc_1p5b_launch.sh \
#       > outputs/exp_mc_1p5b_pilot_launch.log 2>&1 &
#
# After training finishes, each run's final/ is an HF ckpt directory,
# and a platform-ready copy is auto-staged to:
#   outputs/platform_eval_ready/<EXP_NAME_BASE>_<cfg>_s<seed>/global_step<T>/hf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"
STAGE="${STAGE:-pilot}"

# ---- shared inputs (reuse existing artifacts) ----
UTILITY_CSV_MC="${UTILITY_CSV_MC:-outputs/utility_mc_v1/utility_matrix_mc_sum_delta.csv}"
BASE_CLUSTER_IDS="${BASE_CLUSTER_IDS:-outputs/base_hidden_l2rm1pc_real/cluster_ids_initial.npy}"
N_META_CLUSTERS="${N_META_CLUSTERS:-24}"
SEED="${SEED:-42}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B}"

# ---- stage-specific defaults ----
if [[ "${STAGE}" == "pilot" ]]; then
  TOTAL_ITERS_DEFAULT=1000
  EVAL_INTERVAL_DEFAULT=200
  SAVE_INTERVAL_DEFAULT=999999      # save only at end
  PMP_UPDATE_INTERVAL_DEFAULT=50
elif [[ "${STAGE}" == "main" ]]; then
  TOTAL_ITERS_DEFAULT=12000
  EVAL_INTERVAL_DEFAULT=500
  SAVE_INTERVAL_DEFAULT=4000
  PMP_UPDATE_INTERVAL_DEFAULT=100
else
  echo "[error] STAGE must be pilot or main, got ${STAGE}" >&2
  exit 1
fi

TOTAL_ITERS="${TOTAL_ITERS:-${TOTAL_ITERS_DEFAULT}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${EVAL_INTERVAL_DEFAULT}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${SAVE_INTERVAL_DEFAULT}}"
PMP_UPDATE_INTERVAL="${PMP_UPDATE_INTERVAL:-${PMP_UPDATE_INTERVAL_DEFAULT}}"

# ---- training hyperparams ----
MAX_LENGTH="${MAX_LENGTH:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

# ---- layout ----
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME_BASE="exp_mc_1p5b_${STAGE}_m${N_META_CLUSTERS}_t${TOTAL_ITERS}_s${SEED}_${TAG}"
OUT_ROOT="outputs/${EXP_NAME_BASE}"
GPUS=(0 1 2)   # 3 configs -> 3 GPUs in parallel; GPU 3 kept free

# ---- sanity checks ----
if [[ ! -f "${UTILITY_CSV_MC}" ]]; then
  echo "[error] missing ${UTILITY_CSV_MC}" >&2; exit 1
fi
if [[ ! -f "${BASE_CLUSTER_IDS}" ]]; then
  echo "[error] missing ${BASE_CLUSTER_IDS}" >&2; exit 1
fi

mkdir -p "${OUT_ROOT}/meta_clusters" "${OUT_ROOT}/runs" "${OUT_ROOT}/logs"
printf '%s\n' "${OUT_ROOT}" > outputs/latest_exp_mc_1p5b_out_root.txt

cat > "${OUT_ROOT}/launch_manifest.json" <<EOF
{
  "tag": "${TAG}",
  "stage": "${STAGE}",
  "model_path": "${MODEL_PATH}",
  "utility_csv_mc": "${UTILITY_CSV_MC}",
  "base_cluster_ids": "${BASE_CLUSTER_IDS}",
  "n_meta_clusters": ${N_META_CLUSTERS},
  "total_iters": ${TOTAL_ITERS},
  "max_length": ${MAX_LENGTH},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "grad_accum": ${GRAD_ACCUM},
  "tokens_per_step_approx": $((MAX_LENGTH * TRAIN_BATCH_SIZE * GRAD_ACCUM)),
  "tokens_total_approx": $((MAX_LENGTH * TRAIN_BATCH_SIZE * GRAD_ACCUM * TOTAL_ITERS)),
  "pmp_update_interval": ${PMP_UPDATE_INTERVAL},
  "eval_interval": ${EVAL_INTERVAL},
  "save_interval": ${SAVE_INTERVAL},
  "seed": ${SEED},
  "dev_domains": ["math", "code"],
  "configs": ["hidden_base", "hidden_utility_meta", "hidden_random_meta"],
  "gpus": "${GPUS[*]}"
}
EOF

# ---- step 1: build meta-clusters (reuses math+code utility csv) ----
build_meta() {
  local mode="$1"
  local out_dir="${OUT_ROOT}/meta_clusters/${mode}_m${N_META_CLUSTERS}"
  echo "[meta] building ${mode} -> ${out_dir}"
  "${PYTHON_BIN}" analysis/build_meta_clusters.py \
    --utility-csv "${UTILITY_CSV_MC}" \
    --base-cluster-ids "${BASE_CLUSTER_IDS}" \
    --out-dir "${out_dir}" \
    --n-meta-clusters "${N_META_CLUSTERS}" \
    --mode "${mode}" \
    --normalize zscore \
    --seed 42
}
build_meta utility
build_meta random

UTIL_IDS="${OUT_ROOT}/meta_clusters/utility_m${N_META_CLUSTERS}/meta_cluster_ids.npy"
RAND_IDS="${OUT_ROOT}/meta_clusters/random_m${N_META_CLUSTERS}/meta_cluster_ids.npy"

# ---- step 2: launch 3 configs concurrently, one GPU each ----
declare -a JOBS=()
JOBS+=("hidden_base|${BASE_CLUSTER_IDS}")
JOBS+=("hidden_utility_meta|${UTIL_IDS}")
JOBS+=("hidden_random_meta|${RAND_IDS}")

run_job() {
  local idx="$1" entry="$2" gpu="$3"
  IFS='|' read -r cfg precomp <<< "${entry}"
  local out_dir="${OUT_ROOT}/runs/${cfg}_s${SEED}"
  local log="${OUT_ROOT}/logs/job_${idx}_${cfg}_s${SEED}_gpu${gpu}.log"

  if [[ -e "${out_dir}/done.flag" ]]; then
    echo "[skip] ${out_dir} already done"
    return
  fi
  mkdir -p "${out_dir}"

  echo "[start] job=${idx} cfg=${cfg} seed=${SEED} gpu=${gpu} -> ${out_dir}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    RUN_NAME="${cfg}" \
    PRECOMPUTED_IDS="${precomp}" \
    OUT_DIR="${out_dir}" \
    SEED="${SEED}" \
    MODEL_PATH="${MODEL_PATH}" \
    TOTAL_ITERS="${TOTAL_ITERS}" \
    EVAL_INTERVAL="${EVAL_INTERVAL}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    PMP_UPDATE_INTERVAL="${PMP_UPDATE_INTERVAL}" \
    MAX_LENGTH="${MAX_LENGTH}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    GRAD_ACCUM="${GRAD_ACCUM}" \
    DEV_BATCH_SIZE="${DEV_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    LR="${LR}" \
    WARMUP_ITERS="${WARMUP_ITERS}" \
    MASTER_PORT="$((29641 + idx))" \
    bash scripts/run_exp_mc_one.sh \
      && touch "${out_dir}/done.flag"
  ) >"${log}" 2>&1
  echo "[end  ] job=${idx} cfg=${cfg} seed=${SEED} gpu=${gpu} (rc=$?)"
}

idx=0
for entry in "${JOBS[@]}"; do
  gpu="${GPUS[$idx]}"
  run_job "${idx}" "${entry}" "${gpu}" &
  idx=$((idx + 1))
  sleep 2
done
wait

echo "[done] all ${#JOBS[@]} runs finished under ${OUT_ROOT}"

# ---- step 3: auto-stage HF checkpoints for platform eval ----
if [[ "${AUTO_STAGE:-1}" == "1" ]]; then
  echo "[stage] staging HF ckpts for platform eval..."
  for cfg in hidden_base hidden_utility_meta hidden_random_meta; do
    src="${OUT_ROOT}/runs/${cfg}_s${SEED}/final"
    if [[ -d "${src}" ]]; then
      SRC_HF="${src}" \
      DST_ROOT="${REPO_ROOT}/outputs/platform_eval_ready" \
      EXP_NAME="${EXP_NAME_BASE}__${cfg}_s${SEED}" \
      STEP="${TOTAL_ITERS}" \
      COPY_MODE=copy \
      bash scripts/stage_for_platform_eval.sh || echo "[stage] WARN: staging ${cfg} failed"
    else
      echo "[stage] WARN: no final/ for ${cfg}, skip"
    fi
  done
  echo "[stage] done. platform-ready dirs under outputs/platform_eval_ready/"
fi
