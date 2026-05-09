#!/usr/bin/env bash
# Run ONE Exp-mc training run (math+code only).
#
# Required env:
#   RUN_NAME           : hidden_base | hidden_utility_meta | hidden_random_meta
#   PRECOMPUTED_IDS    : path to .npy with cluster ids
#   OUT_DIR            : where to write checkpoints + logs
#   CUDA_VISIBLE_DEVICES : pin to one GPU
#   SEED               : training seed (also propagated to dataloader)
#   MASTER_PORT        : torchrun master port (only matters if NPROC>1)
#
# Optional env (with defaults):
#   TOTAL_ITERS=600
#   N_META_CLUSTERS=24  # only used in OUT_ROOT naming, not in this script
#   PMP_UPDATE_INTERVAL=20
#   EVAL_INTERVAL=100
#   LR=1e-5
#   WARMUP_ITERS=20
#
# This script writes a run_config.yaml with `dev_domains = [math, code]`
# (NO reading) and launches train.py.

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
TOTAL_ITERS="${TOTAL_ITERS:-600}"
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
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29541}"

: "${RUN_NAME:?RUN_NAME required}"
: "${PRECOMPUTED_IDS:?PRECOMPUTED_IDS required}"
: "${OUT_DIR:?OUT_DIR required}"
: "${SEED:?SEED required}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"
RUN_CONFIG="${OUT_DIR}/run_config.yaml"

cat > "${OUT_DIR}/run_metadata.json" <<EOF
{
  "name": "${RUN_NAME}",
  "precomputed_cluster_ids": "${PRECOMPUTED_IDS}",
  "train_dir": "${TRAIN_DIR}",
  "dev_root": "${DEV_ROOT}",
  "dev_domains": ["math", "code"],
  "seed": ${SEED},
  "total_iters": ${TOTAL_ITERS}
}
EOF

# --- write run_config.yaml (math + code dev only, NO reading) ---
RUN_CONFIG_PATH="${RUN_CONFIG}" \
BASE_CONFIG_PATH="${CONFIG}" \
PRECOMPUTED_IDS_PATH="${PRECOMPUTED_IDS}" \
MODEL_PATH_VALUE="${MODEL_PATH}" \
TRAIN_DIR_VALUE="${TRAIN_DIR}" \
DEV_ROOT_VALUE="${DEV_ROOT}" \
TEXT_FIELD_VALUE="${TEXT_FIELD}" \
DEV_NUM_VALUE="${DEV_NUM}" \
DEV_SEED_VALUE="${DEV_SEED}" \
TOTAL_ITERS_VALUE="${TOTAL_ITERS}" \
TRAIN_BATCH_SIZE_VALUE="${TRAIN_BATCH_SIZE}" \
GRAD_ACCUM_VALUE="${GRAD_ACCUM}" \
EVAL_BATCH_SIZE_VALUE="${EVAL_BATCH_SIZE}" \
EVAL_INTERVAL_VALUE="${EVAL_INTERVAL}" \
SAVE_INTERVAL_VALUE="${SAVE_INTERVAL}" \
OUT_DIR_VALUE="${OUT_DIR}" \
LR_VALUE="${LR}" \
WARMUP_ITERS_VALUE="${WARMUP_ITERS}" \
SEED_VALUE="${SEED}" \
PMP_UPDATE_INTERVAL_VALUE="${PMP_UPDATE_INTERVAL}" \
DEV_BATCH_SIZE_VALUE="${DEV_BATCH_SIZE}" \
MAX_LENGTH_VALUE="${MAX_LENGTH}" \
"${PYTHON_BIN}" - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["BASE_CONFIG_PATH"])
cfg.model.path = os.environ["MODEL_PATH_VALUE"]
cfg.model.attn_impl = "sdpa"
cfg.model.max_length = int(os.environ["MAX_LENGTH_VALUE"])
cfg.model.gradient_checkpointing = False
cfg.data.train_dir = os.environ["TRAIN_DIR_VALUE"]
# Use math as the "primary" dev_dir for backward compat fields, but the
# real eval lives in dev_domains below (math + code only).
cfg.data.dev_dir = f"{os.environ['DEV_ROOT_VALUE']}/math"
cfg.data.dev_domains = [
    {"name": "math", "dir": f"{os.environ['DEV_ROOT_VALUE']}/math", "weight": 1.0},
    {"name": "code", "dir": f"{os.environ['DEV_ROOT_VALUE']}/code", "weight": 1.0},
]
cfg.data.text_field = os.environ["TEXT_FIELD_VALUE"]
cfg.data.dev_num = int(os.environ["DEV_NUM_VALUE"])
cfg.data.dev_seed = int(os.environ["DEV_SEED_VALUE"])
cfg.data.eval_format = "text"
cfg.data.num_workers = 0
cfg.training.total_iters = int(os.environ["TOTAL_ITERS_VALUE"])
cfg.training.batch_size = int(os.environ["TRAIN_BATCH_SIZE_VALUE"])
cfg.training.gradient_accumulation_steps = int(os.environ["GRAD_ACCUM_VALUE"])
cfg.training.eval_batch_size = int(os.environ["EVAL_BATCH_SIZE_VALUE"])
cfg.training.eval_interval = int(os.environ["EVAL_INTERVAL_VALUE"])
cfg.training.save_interval = int(os.environ["SAVE_INTERVAL_VALUE"])
cfg.training.no_eval_at_start = False
cfg.training.save_dir = os.environ["OUT_DIR_VALUE"]
cfg.training.lr = float(os.environ["LR_VALUE"])
cfg.training.warmup_iters = int(os.environ["WARMUP_ITERS_VALUE"])
cfg.training.seed = int(os.environ["SEED_VALUE"])
cfg.clustering.precomputed_ids_path = os.environ["PRECOMPUTED_IDS_PATH"]
cfg.clustering.recluster_interval = -1
cfg.pmp.update_interval = int(os.environ["PMP_UPDATE_INTERVAL_VALUE"])
cfg.pmp.dev_batch_size = int(os.environ["DEV_BATCH_SIZE_VALUE"])
cfg.pmp.drop_bad_clusters = False
# Make sure no utility_prior is left over from earlier Exp4 configs.
if hasattr(cfg.pmp, "utility_prior"):
    cfg.pmp.utility_prior.enabled = False
cfg.deepspeed.enabled = False
OmegaConf.save(cfg, os.environ["RUN_CONFIG_PATH"])
print(f"[config] wrote {os.environ['RUN_CONFIG_PATH']}")
PY

echo
echo "============================================================"
echo "[run] ${RUN_NAME}  (seed=${SEED})  GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[out] ${OUT_DIR}"
echo "[cluster] ${PRECOMPUTED_IDS}"
echo "============================================================"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
      train.py --config "${RUN_CONFIG}" 2>&1 | tee "${OUT_DIR}/train.log"
else
  "${PYTHON_BIN}" train.py --config "${RUN_CONFIG}" 2>&1 | tee "${OUT_DIR}/train.log"
fi
