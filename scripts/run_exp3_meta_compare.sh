#!/usr/bin/env bash
# Run Experiment 3 comparisons using meta-clusters derived from Exp2 utility vectors.
#
# By default this script:
#   1) builds utility-driven meta-clusters (z-score + KMeans)
#   2) builds random-regroup meta-clusters
#   3) runs the same training pipeline on:
#        - hidden_base
#        - hidden_utility_meta
#        - hidden_random_meta
#
# Launch example:
#   nohup bash scripts/run_exp3_meta_compare.sh > outputs/exp3_pipeline.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/torchrun}"
CONFIG="${CONFIG:-configs/default.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
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
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29531}"
N_META_CLUSTERS="${N_META_CLUSTERS:-16}"
CLUSTER_LABEL="${CLUSTER_LABEL:-hidden_l2rm1pc_real}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/exp3_${CLUSTER_LABEL}_m${N_META_CLUSTERS}_${RUN_TAG}}"
BASE_CLUSTER_IDS="${BASE_CLUSTER_IDS:-outputs/base_hidden_l2rm1pc_real/cluster_ids_initial.npy}"
LATEST_EXP2_PTR="${LATEST_EXP2_PTR:-outputs/latest_exp2_real_out_root.txt}"
UTILITY_CSV="${UTILITY_CSV:-}"
RUN_BASE_HIDDEN="${RUN_BASE_HIDDEN:-1}"
RUN_UTILITY_META="${RUN_UTILITY_META:-1}"
RUN_RANDOM_META="${RUN_RANDOM_META:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${UTILITY_CSV}" && -f "${LATEST_EXP2_PTR}" ]]; then
  latest_exp2_root="$(cat "${LATEST_EXP2_PTR}")"
  UTILITY_CSV="${latest_exp2_root}/utility_matrix/utility_matrix_sum_delta.csv"
fi

if [[ ! -f "${BASE_CLUSTER_IDS}" ]]; then
  echo "[error] BASE_CLUSTER_IDS not found: ${BASE_CLUSTER_IDS}"
  exit 1
fi

if [[ -z "${UTILITY_CSV}" || ! -f "${UTILITY_CSV}" ]]; then
  echo "[error] UTILITY_CSV not found: ${UTILITY_CSV}"
  exit 1
fi

for ability in math reading code; do
  if [[ ! -d "${DEV_ROOT}/${ability}" ]]; then
    echo "[error] Missing dev dir: ${DEV_ROOT}/${ability}"
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/meta_clusters" "${OUT_ROOT}/runs"
printf '%s\n' "${OUT_ROOT}" > outputs/latest_exp3_out_root.txt

cat > "${OUT_ROOT}/run_manifest.json" <<EOF
{
  "cluster_label": "${CLUSTER_LABEL}",
  "base_cluster_ids": "${BASE_CLUSTER_IDS}",
  "utility_csv": "${UTILITY_CSV}",
  "train_dir": "${TRAIN_DIR}",
  "dev_root": "${DEV_ROOT}",
  "n_meta_clusters": ${N_META_CLUSTERS},
  "seed": ${SEED},
  "total_iters": ${TOTAL_ITERS}
}
EOF

build_meta() {
  local mode="$1"
  local out_dir="${OUT_ROOT}/meta_clusters/${mode}_m${N_META_CLUSTERS}"
  echo "[meta] building ${mode} -> ${out_dir}"
  "${PYTHON_BIN}" analysis/build_meta_clusters.py \
    --utility-csv "${UTILITY_CSV}" \
    --base-cluster-ids "${BASE_CLUSTER_IDS}" \
    --out-dir "${out_dir}" \
    --n-meta-clusters "${N_META_CLUSTERS}" \
    --mode "${mode}" \
    --normalize zscore \
    --seed "${SEED}"
}

build_meta utility
build_meta random

DEV_DOMAINS_OVERRIDE="data.dev_domains=[{name:math,dir:${DEV_ROOT}/math,weight:1.0},{name:reading,dir:${DEV_ROOT}/reading,weight:1.0},{name:code,dir:${DEV_ROOT}/code,weight:1.0}]"

run_one() {
  local name="$1"
  local precomputed_ids_path="$2"
  local out_dir="${OUT_ROOT}/runs/${name}"

  if [[ -e "${out_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "[error] Output directory already exists: ${out_dir}"
    echo "        Set ALLOW_EXISTING=1 to reuse it, or set a new OUT_ROOT."
    exit 1
  fi
  mkdir -p "${out_dir}"

  cat > "${out_dir}/run_metadata.json" <<EOF
{
  "name": "${name}",
  "precomputed_cluster_ids": "${precomputed_ids_path}",
  "train_dir": "${TRAIN_DIR}",
  "dev_root": "${DEV_ROOT}",
  "n_meta_clusters": ${N_META_CLUSTERS},
  "seed": ${SEED},
  "total_iters": ${TOTAL_ITERS}
}
EOF

  echo
  echo "============================================================"
  echo "[run] ${name}"
  echo "[out] ${out_dir}"
  echo "[cluster] ${precomputed_ids_path}"
  echo "[launch] nproc=${NPROC_PER_NODE}"
  echo "============================================================"

  local -a cmd
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    cmd=(
      "${TORCHRUN_BIN}"
      --nproc_per_node="${NPROC_PER_NODE}"
      --master_port="${MASTER_PORT}"
      train.py
    )
  else
    cmd=("${PYTHON_BIN}" train.py)
  fi

  "${cmd[@]}" --config "${CONFIG}" \
    model.path="${MODEL_PATH}" \
    model.attn_impl=sdpa \
    model.max_length="${MAX_LENGTH}" \
    model.gradient_checkpointing=false \
    data.train_dir="${TRAIN_DIR}" \
    data.dev_dir="${DEV_ROOT}/math" \
    "${DEV_DOMAINS_OVERRIDE}" \
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
    clustering.precomputed_ids_path="${precomputed_ids_path}" \
    clustering.recluster_interval=-1 \
    pmp.update_interval="${PMP_UPDATE_INTERVAL}" \
    pmp.dev_batch_size="${DEV_BATCH_SIZE}" \
    pmp.drop_bad_clusters=false \
    deepspeed.enabled=false \
    2>&1 | tee "${out_dir}/train.log"
}

if [[ "${RUN_BASE_HIDDEN}" == "1" ]]; then
  run_one "hidden_base" "${BASE_CLUSTER_IDS}"
fi
if [[ "${RUN_UTILITY_META}" == "1" ]]; then
  run_one "hidden_utility_meta" "${OUT_ROOT}/meta_clusters/utility_m${N_META_CLUSTERS}/meta_cluster_ids.npy"
fi
if [[ "${RUN_RANDOM_META}" == "1" ]]; then
  run_one "hidden_random_meta" "${OUT_ROOT}/meta_clusters/random_m${N_META_CLUSTERS}/meta_cluster_ids.npy"
fi

echo
echo "[done] Exp3 runs finished. Outputs: ${OUT_ROOT}"
