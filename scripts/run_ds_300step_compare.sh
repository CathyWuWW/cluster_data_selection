#!/usr/bin/env bash
# Run a comparable 300-step data-selection sweep:
#   1) random clusters
#   2) raw intermediate hidden-mean clusters
#   3) L2-normalized + remove-1-PC intermediate hidden-mean clusters
#
# The script first materializes a fixed 256-sample dev set so all runs evaluate
# on exactly the same examples.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
CONFIG="${CONFIG:-configs/default.yaml}"
TRAIN_DIR="${TRAIN_DIR:-local_data/slimpajama_6b_balanced/train}"
VALID_DIR="${VALID_DIR:-valid}"
TEXT_FIELD="${TEXT_FIELD:-text}"
DEV_NUM="${DEV_NUM:-256}"
DEV_SEED="${DEV_SEED:-42}"
FIXED_DEV_DIR="${FIXED_DEV_DIR:-local_data/fixed_dev_${DEV_NUM}_seed${DEV_SEED}}"

TOTAL_ITERS="${TOTAL_ITERS:-300}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
PMP_UPDATE_INTERVAL="${PMP_UPDATE_INTERVAL:-20}"
CLUSTER_SIZE="${CLUSTER_SIZE:-1000}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DEV_BATCH_SIZE="${DEV_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-20}"
SEED="${SEED:-42}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/ds_compare_300step_dev${DEV_NUM}_${RUN_TAG}}"

prepare_fixed_dev() {
  mkdir -p "${FIXED_DEV_DIR}"
  local out_file="${FIXED_DEV_DIR}/dev_fixed_${DEV_NUM}_seed${DEV_SEED}.jsonl"
  if [[ -s "${out_file}" ]] && [[ "$(wc -l < "${out_file}")" -eq "${DEV_NUM}" ]]; then
    echo "[dev] Reusing fixed dev set: ${out_file}"
    return
  fi

  echo "[dev] Creating fixed dev set: ${out_file}"
  VALID_DIR="${VALID_DIR}" \
  TEXT_FIELD="${TEXT_FIELD}" \
  DEV_NUM="${DEV_NUM}" \
  DEV_SEED="${DEV_SEED}" \
  OUT_FILE="${out_file}" \
  python - <<'PY'
import glob
import json
import os
import random

valid_dir = os.environ["VALID_DIR"]
text_field = os.environ["TEXT_FIELD"]
dev_num = int(os.environ["DEV_NUM"])
seed = int(os.environ["DEV_SEED"])
out_file = os.environ["OUT_FILE"]

paths = []
for pattern in ("*.jsonl", "*.json", "**/*.jsonl", "**/*.json"):
    paths.extend(glob.glob(os.path.join(valid_dir, pattern), recursive=True))
paths = sorted(set(paths))
if not paths:
    raise FileNotFoundError(f"No JSON/JSONL files found in {valid_dir}")

records = []
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and text_field in obj:
                    records.append(obj)
        else:
            data = json.load(f)
            if isinstance(data, list):
                records.extend([x for x in data if isinstance(x, dict) and text_field in x])
            elif isinstance(data, dict):
                if text_field in data:
                    records.append(data)
                else:
                    for key in ("data", "items", "records", "samples"):
                        value = data.get(key)
                        if isinstance(value, list):
                            records.extend([x for x in value if isinstance(x, dict) and text_field in x])
                            break

if len(records) < dev_num:
    raise ValueError(f"Need {dev_num} dev records, found {len(records)} in {valid_dir}")

rng = random.Random(seed)
selected = rng.sample(records, dev_num)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    for obj in selected:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(selected)} records to {out_file}")
PY
}

run_one() {
  local name="$1"
  shift
  local out_dir="${OUT_ROOT}/${name}"

  if [[ -e "${out_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "[error] Output directory already exists: ${out_dir}"
    echo "        Set ALLOW_EXISTING=1 to reuse it, or set a new OUT_ROOT."
    exit 1
  fi
  mkdir -p "${out_dir}"

  echo
  echo "============================================================"
  echo "[run] ${name}"
  echo "[out] ${out_dir}"
  echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  python train.py --config "${CONFIG}" \
    model.path="${MODEL_PATH}" \
    model.attn_impl=sdpa \
    model.max_length="${MAX_LENGTH}" \
    model.gradient_checkpointing=false \
    data.train_dir="${TRAIN_DIR}" \
    data.dev_dir="${FIXED_DEV_DIR}" \
    data.text_field="${TEXT_FIELD}" \
    data.dev_num="${DEV_NUM}" \
    data.eval_format=text \
    data.num_workers=0 \
    training.total_iters="${TOTAL_ITERS}" \
    training.batch_size="${TRAIN_BATCH_SIZE}" \
    training.gradient_accumulation_steps="${GRAD_ACCUM}" \
    training.eval_batch_size="${EVAL_BATCH_SIZE}" \
    training.eval_interval="${EVAL_INTERVAL}" \
    training.save_interval=999999 \
    training.no_eval_at_start=false \
    training.save_dir="${out_dir}" \
    training.lr="${LR}" \
    training.warmup_iters="${WARMUP_ITERS}" \
    training.seed="${SEED}" \
    clustering.cluster_size="${CLUSTER_SIZE}" \
    pmp.update_interval="${PMP_UPDATE_INTERVAL}" \
    pmp.dev_batch_size="${DEV_BATCH_SIZE}" \
    pmp.ghost_ip.proj_dim=1024 \
    pmp.drop_bad_clusters=false \
    deepspeed.enabled=false \
    "$@" \
    2>&1 | tee "${out_dir}/train.log"
}

prepare_fixed_dev
mkdir -p "${OUT_ROOT}"

cat > "${OUT_ROOT}/run_config.txt" <<EOF
MODEL_PATH=${MODEL_PATH}
CONFIG=${CONFIG}
TRAIN_DIR=${TRAIN_DIR}
FIXED_DEV_DIR=${FIXED_DEV_DIR}
DEV_NUM=${DEV_NUM}
DEV_SEED=${DEV_SEED}
TOTAL_ITERS=${TOTAL_ITERS}
EVAL_INTERVAL=${EVAL_INTERVAL}
PMP_UPDATE_INTERVAL=${PMP_UPDATE_INTERVAL}
CLUSTER_SIZE=${CLUSTER_SIZE}
MAX_LENGTH=${MAX_LENGTH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SEED=${SEED}
EOF

run_one "random" \
  clustering.method=random \
  clustering.embedding_model.enabled=false \
  clustering.kmeans.normalize=none \
  clustering.kmeans.remove_top_pcs=0 \
  clustering.kmeans.feature_batch_size=256

run_one "raw_intermediate" \
  clustering.method=minibatch \
  clustering.kmeans.feature=intermediate \
  clustering.kmeans.embed_layer=-1 \
  clustering.kmeans.normalize=none \
  clustering.kmeans.remove_top_pcs=0 \
  clustering.kmeans.feature_batch_size=32 \
  clustering.embedding_model.enabled=true \
  clustering.embedding_model.path="${MODEL_PATH}" \
  clustering.embedding_model.dtype=bfloat16 \
  clustering.embedding_model.attn_impl=sdpa

run_one "l2_rm1pc" \
  clustering.method=minibatch \
  clustering.kmeans.feature=intermediate \
  clustering.kmeans.embed_layer=-1 \
  clustering.kmeans.normalize=l2 \
  clustering.kmeans.remove_top_pcs=1 \
  clustering.kmeans.feature_batch_size=32 \
  clustering.embedding_model.enabled=true \
  clustering.embedding_model.path="${MODEL_PATH}" \
  clustering.embedding_model.dtype=bfloat16 \
  clustering.embedding_model.attn_impl=sdpa

echo
echo "[done] All runs finished. Outputs: ${OUT_ROOT}"
