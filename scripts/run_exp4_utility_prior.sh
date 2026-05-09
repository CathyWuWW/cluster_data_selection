#!/usr/bin/env bash
# Run Experiment 4: keep the original 87-way hidden clusters and inject an
# Exp2 utility prior into PMP grad_gamma initialization.
#
# Compared to Exp3, this script does NOT regroup base clusters into m
# meta-clusters. Instead it sweeps the prior strength `alpha` while keeping
# the 87-way structure intact.
#
# Each invocation produces multiple sibling runs under one timestamped root:
#   runs/prior_alpha0   (no prior, sanity check vs. existing base run)
#   runs/prior_alpha0p5
#   runs/prior_alpha1
#   runs/prior_alpha2
#
# Required env (with sensible defaults):
#   BASE_CLUSTER_IDS  path to outputs/base_hidden_l2rm1pc_real/cluster_ids_initial.npy
#   UTILITY_CSV       path to outputs/.../utility_matrix/utility_matrix_sum_delta.csv
#                     If empty, will resolve from outputs/latest_exp2_real_out_root.txt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
PMP_MIN_WEIGHT="${PMP_MIN_WEIGHT:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29541}"

CLUSTER_LABEL="${CLUSTER_LABEL:-hidden_l2rm1pc_real}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/exp4_${CLUSTER_LABEL}_prior_${RUN_TAG}}"
BASE_CLUSTER_IDS="${BASE_CLUSTER_IDS:-outputs/base_hidden_l2rm1pc_real/cluster_ids_initial.npy}"
LATEST_EXP2_PTR="${LATEST_EXP2_PTR:-outputs/latest_exp2_real_out_root.txt}"
UTILITY_CSV="${UTILITY_CSV:-}"

# Prior knobs (defaults follow the analysis we did on Exp2 utility matrix).
PRIOR_AGG="${PRIOR_AGG:-weighted_sum}"      # sum | mean | max | weighted_sum
PRIOR_NORMALIZE="${PRIOR_NORMALIZE:-zscore}"
PRIOR_SIGN="${PRIOR_SIGN:-positive}"
PRIOR_ABILITIES="${PRIOR_ABILITIES:-math,reading,code}"
# Down-weight the noisier reading channel by default.
PRIOR_WEIGHTS="${PRIOR_WEIGHTS:-1.0,0.3,1.0}"
# alpha sweep; alpha=0 is automatically included as a sanity baseline.
PRIOR_ALPHAS="${PRIOR_ALPHAS:-0,0.5,1.0,2.0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${UTILITY_CSV}" && -f "${LATEST_EXP2_PTR}" ]]; then
  latest_exp2_root="$(cat "${LATEST_EXP2_PTR}")"
  UTILITY_CSV="${latest_exp2_root}/utility_matrix/utility_matrix_sum_delta.csv"
fi
if [[ ! -f "${BASE_CLUSTER_IDS}" ]]; then
  echo "[error] BASE_CLUSTER_IDS not found: ${BASE_CLUSTER_IDS}"; exit 1
fi
if [[ ! -f "${UTILITY_CSV}" ]]; then
  echo "[error] UTILITY_CSV not found: ${UTILITY_CSV}"; exit 1
fi
for ability in math reading code; do
  if [[ ! -d "${DEV_ROOT}/${ability}" ]]; then
    echo "[error] Missing dev dir: ${DEV_ROOT}/${ability}"; exit 1
  fi
done

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/runs"
printf '%s\n' "${OUT_ROOT}" > outputs/latest_exp4_out_root.txt

cat > "${OUT_ROOT}/run_manifest.json" <<EOF
{
  "cluster_label": "${CLUSTER_LABEL}",
  "base_cluster_ids": "${BASE_CLUSTER_IDS}",
  "utility_csv": "${UTILITY_CSV}",
  "train_dir": "${TRAIN_DIR}",
  "dev_root": "${DEV_ROOT}",
  "seed": ${SEED},
  "total_iters": ${TOTAL_ITERS},
  "prior_aggregator": "${PRIOR_AGG}",
  "prior_normalize": "${PRIOR_NORMALIZE}",
  "prior_sign": "${PRIOR_SIGN}",
  "prior_abilities": "${PRIOR_ABILITIES}",
  "prior_weights": "${PRIOR_WEIGHTS}",
  "prior_alphas": "${PRIOR_ALPHAS}"
}
EOF

build_run_config() {
  local run_config_path="$1"
  local prior_alpha="$2"

  RUN_CONFIG_PATH="${run_config_path}" \
  BASE_CONFIG_PATH="${CONFIG}" \
  PRECOMPUTED_IDS_PATH="${BASE_CLUSTER_IDS}" \
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
  OUT_DIR_VALUE="$(dirname "${run_config_path}")" \
  LR_VALUE="${LR}" \
  WARMUP_ITERS_VALUE="${WARMUP_ITERS}" \
  SEED_VALUE="${SEED}" \
  PMP_UPDATE_INTERVAL_VALUE="${PMP_UPDATE_INTERVAL}" \
  PMP_MIN_WEIGHT_VALUE="${PMP_MIN_WEIGHT}" \
  DEV_BATCH_SIZE_VALUE="${DEV_BATCH_SIZE}" \
  MAX_LENGTH_VALUE="${MAX_LENGTH}" \
  PRIOR_ENABLED_VALUE="$( [[ "${prior_alpha}" == "0" || "${prior_alpha}" == "0.0" ]] && echo False || echo True )" \
  PRIOR_CSV_VALUE="${UTILITY_CSV}" \
  PRIOR_AGG_VALUE="${PRIOR_AGG}" \
  PRIOR_NORM_VALUE="${PRIOR_NORMALIZE}" \
  PRIOR_SIGN_VALUE="${PRIOR_SIGN}" \
  PRIOR_ALPHA_VALUE="${prior_alpha}" \
  PRIOR_ABILITIES_VALUE="${PRIOR_ABILITIES}" \
  PRIOR_WEIGHTS_VALUE="${PRIOR_WEIGHTS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["BASE_CONFIG_PATH"])
cfg.model.path = os.environ["MODEL_PATH_VALUE"]
cfg.model.attn_impl = "sdpa"
cfg.model.max_length = int(os.environ["MAX_LENGTH_VALUE"])
cfg.model.gradient_checkpointing = False
cfg.data.train_dir = os.environ["TRAIN_DIR_VALUE"]
dev_root = os.environ["DEV_ROOT_VALUE"]
cfg.data.dev_dir = f"{dev_root}/math"
cfg.data.dev_domains = [
    {"name": "math",    "dir": f"{dev_root}/math",    "weight": 1.0},
    {"name": "reading", "dir": f"{dev_root}/reading", "weight": 1.0},
    {"name": "code",    "dir": f"{dev_root}/code",    "weight": 1.0},
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
cfg.pmp.min_weight = float(os.environ["PMP_MIN_WEIGHT_VALUE"])
cfg.pmp.dev_batch_size = int(os.environ["DEV_BATCH_SIZE_VALUE"])
cfg.pmp.drop_bad_clusters = False
cfg.deepspeed.enabled = False

# Inject utility_prior block (overrides defaults from configs/default.yaml).
abilities = [a for a in os.environ["PRIOR_ABILITIES_VALUE"].split(",") if a]
weights = [float(x) for x in os.environ["PRIOR_WEIGHTS_VALUE"].split(",") if x]
cfg.pmp.utility_prior = {
    "enabled": os.environ["PRIOR_ENABLED_VALUE"] == "True",
    "csv_path": os.environ["PRIOR_CSV_VALUE"],
    "abilities": abilities,
    "weights": weights,
    "aggregator": os.environ["PRIOR_AGG_VALUE"],
    "normalize": os.environ["PRIOR_NORM_VALUE"],
    "alpha": float(os.environ["PRIOR_ALPHA_VALUE"]),
    "sign": os.environ["PRIOR_SIGN_VALUE"],
}
OmegaConf.save(cfg, os.environ["RUN_CONFIG_PATH"])
print(f"[config] wrote {os.environ['RUN_CONFIG_PATH']}")
PY
}

run_one() {
  local alpha="$1"
  local tag="$2"
  local out_dir="${OUT_ROOT}/runs/${tag}"
  local run_config="${out_dir}/run_config.yaml"
  if [[ -e "${out_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "[error] Output directory already exists: ${out_dir}"
    echo "        Set ALLOW_EXISTING=1 to reuse it, or set a new OUT_ROOT."
    exit 1
  fi
  mkdir -p "${out_dir}"

  cat > "${out_dir}/run_metadata.json" <<EOF
{
  "name": "${tag}",
  "alpha": ${alpha},
  "precomputed_cluster_ids": "${BASE_CLUSTER_IDS}",
  "utility_csv": "${UTILITY_CSV}",
  "prior_aggregator": "${PRIOR_AGG}",
  "prior_normalize": "${PRIOR_NORMALIZE}",
  "prior_sign": "${PRIOR_SIGN}",
  "prior_abilities": "${PRIOR_ABILITIES}",
  "prior_weights": "${PRIOR_WEIGHTS}",
  "seed": ${SEED},
  "total_iters": ${TOTAL_ITERS}
}
EOF

  build_run_config "${run_config}" "${alpha}"

  echo
  echo "============================================================"
  echo "[run]   ${tag}  (alpha=${alpha})"
  echo "[out]   ${out_dir}"
  echo "[cfg]   ${run_config}"
  echo "[base]  ${BASE_CLUSTER_IDS}"
  echo "[util]  ${UTILITY_CSV}"
  echo "[gpu]   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
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
  "${cmd[@]}" --config "${run_config}" 2>&1 | tee "${out_dir}/train.log"
}

# Convert PRIOR_ALPHAS into "alphaXpY" sibling tags.
IFS=',' read -r -a ALPHA_LIST <<< "${PRIOR_ALPHAS}"
for alpha in "${ALPHA_LIST[@]}"; do
  alpha_trimmed="$(echo "${alpha}" | xargs)"
  if [[ -z "${alpha_trimmed}" ]]; then continue; fi
  tag="prior_alpha$(echo "${alpha_trimmed}" | sed 's/\./p/g')"
  run_one "${alpha_trimmed}" "${tag}"
done

echo
echo "[done] All Exp4 prior runs finished under ${OUT_ROOT}"
