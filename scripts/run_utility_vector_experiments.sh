#!/usr/bin/env bash
# Run Experiment 2 utility-vector training jobs on fixed precomputed clusters.
#
# Expected dev layout:
#   ${DEV_ROOT}/math
#   ${DEV_ROOT}/reading
#   ${DEV_ROOT}/code
#
# Example:
#   PRECOMPUTED_CLUSTER_IDS=outputs/base_hidden/cluster_ids_initial.npy \
#   DEV_ROOT=data/dev \
#   NPROC_PER_NODE=4 \
#   bash scripts/run_utility_vector_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/dataset-assist-0/miniconda/bin/activate}"
CONDA_ENV="${CONDA_ENV:-Data_selection}"

if [[ -f "${CONDA_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE}"
  conda activate "${CONDA_ENV}"
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/dataset-assist-0/usr/lh/wyn/data_selection/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
CONFIG="${CONFIG:-configs/default.yaml}"
TRAIN_DIR="${TRAIN_DIR:-local_data/slimpajama_6b_balanced/train}"
DEV_ROOT="${DEV_ROOT:-data/dev}"
TEXT_FIELD="${TEXT_FIELD:-text}"
DEV_NUM="${DEV_NUM:-256}"
DEV_SEED="${DEV_SEED:-42}"

TOTAL_ITERS="${TOTAL_ITERS:-300}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-999999}"
PMP_UPDATE_INTERVAL="${PMP_UPDATE_INTERVAL:-20}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
SEED="${SEED:-42}"

PRECOMPUTED_CLUSTER_IDS="${PRECOMPUTED_CLUSTER_IDS:-}"
CLUSTER_LABEL="${CLUSTER_LABEL:-hidden_l2rm1pc}"
ABILITIES="${ABILITIES:-math,reading,code}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/utility_vector_${CLUSTER_LABEL}_${RUN_TAG}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${PRECOMPUTED_CLUSTER_IDS}" ]]; then
  echo "[error] PRECOMPUTED_CLUSTER_IDS is required."
  echo "        Example: PRECOMPUTED_CLUSTER_IDS=outputs/base_hidden/cluster_ids_initial.npy"
  exit 1
fi

if [[ ! -f "${PRECOMPUTED_CLUSTER_IDS}" ]]; then
  echo "[error] PRECOMPUTED_CLUSTER_IDS not found: ${PRECOMPUTED_CLUSTER_IDS}"
  exit 1
fi

IFS=',' read -r -a ABILITY_LIST <<< "${ABILITIES}"
mkdir -p "${OUT_ROOT}"

cat > "${OUT_ROOT}/run_manifest.json" <<EOF
{
  "cluster_label": "${CLUSTER_LABEL}",
  "precomputed_cluster_ids": "${PRECOMPUTED_CLUSTER_IDS}",
  "train_dir": "${TRAIN_DIR}",
  "dev_root": "${DEV_ROOT}",
  "abilities": "${ABILITIES}",
  "model_path": "${MODEL_PATH}",
  "seed": ${SEED},
  "dev_seed": ${DEV_SEED},
  "total_iters": ${TOTAL_ITERS},
  "pmp_update_interval": ${PMP_UPDATE_INTERVAL}
}
EOF

run_one() {
  local ability="$1"
  local dev_dir="${DEV_ROOT}/${ability}"
  local out_dir="${OUT_ROOT}/${ability}"

  if [[ ! -d "${dev_dir}" ]]; then
    echo "[error] Ability dev dir not found: ${dev_dir}"
    exit 1
  fi

  if [[ -e "${out_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "[error] Output directory already exists: ${out_dir}"
    echo "        Set ALLOW_EXISTING=1 to reuse it, or set a new OUT_ROOT."
    exit 1
  fi
  mkdir -p "${out_dir}"

  cat > "${out_dir}/run_metadata.json" <<EOF
{
  "ability": "${ability}",
  "cluster_label": "${CLUSTER_LABEL}",
  "precomputed_cluster_ids": "${PRECOMPUTED_CLUSTER_IDS}",
  "train_dir": "${TRAIN_DIR}",
  "dev_dir": "${dev_dir}",
  "seed": ${SEED},
  "dev_seed": ${DEV_SEED},
  "total_iters": ${TOTAL_ITERS},
  "eval_interval": ${EVAL_INTERVAL},
  "pmp_update_interval": ${PMP_UPDATE_INTERVAL}
}
EOF

  echo
  echo "============================================================"
  echo "[run] ${ability}"
  echo "[out] ${out_dir}"
  echo "[dev] ${dev_dir}"
  echo "[cluster] ${PRECOMPUTED_CLUSTER_IDS}"
  echo "[launch] nproc=${NPROC_PER_NODE}"
  echo "============================================================"

  local -a cmd
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    cmd=(
      torchrun
      --nproc_per_node="${NPROC_PER_NODE}"
      --master_port="${MASTER_PORT}"
      train.py
    )
  else
    cmd=(python train.py)
  fi

  "${cmd[@]}" --config "${CONFIG}" \
    model.path="${MODEL_PATH}" \
    model.attn_impl=sdpa \
    model.max_length="${MAX_LENGTH}" \
    model.gradient_checkpointing=false \
    data.train_dir="${TRAIN_DIR}" \
    data.dev_dir="${dev_dir}" \
    data.dev_domains=[] \
    data.text_field="${TEXT_FIELD}" \
    data.dev_num="${DEV_NUM}" \
    data.dev_seed="${DEV_SEED}" \
    data.eval_format=text \
    data.num_workers=0 \
    training.total_iters="${TOTAL_ITERS}" \
    training.batch_size="${TRAIN_BATCH_SIZE}" \
    training.gradient_accumulation_steps="${GRAD_ACCUM}" \
    training.eval_batch_size="${EVAL_BATCH_SIZE}" \
    training.eval_interval="${EVAL_INTERVAL}" \
    training.save_interval="${SAVE_INTERVAL}" \
    training.no_eval_at_start=false \
    training.save_dir="${out_dir}" \
    training.lr="${LR}" \
    training.warmup_iters="${WARMUP_ITERS}" \
    training.seed="${SEED}" \
    clustering.precomputed_ids_path="${PRECOMPUTED_CLUSTER_IDS}" \
    clustering.recluster_interval=-1 \
    pmp.update_interval="${PMP_UPDATE_INTERVAL}" \
    pmp.dev_batch_size="${DEV_BATCH_SIZE}" \
    pmp.drop_bad_clusters=false \
    deepspeed.enabled=false \
    2>&1 | tee "${out_dir}/train.log"
}

for ability in "${ABILITY_LIST[@]}"; do
  run_one "${ability}"
done

echo
echo "[done] Utility runs finished. Outputs: ${OUT_ROOT}"
