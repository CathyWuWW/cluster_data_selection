#!/usr/bin/env bash
# Stage an existing HF checkpoint into the platform companion pretrain-eval layout:
#   <dst_root>/<exp_name>/global_step<step>/hf/
#
# Required:
#   SRC_HF      existing HF checkpoint dir, e.g. outputs/.../final
#   DST_ROOT    staging root, e.g. outputs/platform_eval_ready
#   EXP_NAME    platform experiment name
#   STEP        integer step, e.g. 600
#
# Optional:
#   COPY_MODE=copy|symlink  (default: copy; symlink saves space but may not work
#                            if the platform cannot resolve local paths)

set -euo pipefail

: "${SRC_HF:?SRC_HF required}"
: "${DST_ROOT:?DST_ROOT required}"
: "${EXP_NAME:?EXP_NAME required}"
: "${STEP:?STEP required}"

COPY_MODE="${COPY_MODE:-copy}"
DST="${DST_ROOT}/${EXP_NAME}/global_step${STEP}/hf"

if [[ ! -d "${SRC_HF}" ]]; then
  echo "[error] SRC_HF not found: ${SRC_HF}" >&2
  exit 1
fi

for f in config.json tokenizer.json tokenizer_config.json; do
  if [[ ! -f "${SRC_HF}/${f}" ]]; then
    echo "[error] missing required file: ${SRC_HF}/${f}" >&2
    exit 1
  fi
done
if ! compgen -G "${SRC_HF}/model*.safetensors" >/dev/null && ! compgen -G "${SRC_HF}/pytorch_model*.bin" >/dev/null; then
  echo "[error] no model weights found under ${SRC_HF}" >&2
  exit 1
fi

mkdir -p "$(dirname "${DST}")"
rm -rf "${DST}"

case "${COPY_MODE}" in
  copy)
    mkdir -p "${DST}"
    cp -a "${SRC_HF}/." "${DST}/"
    ;;
  symlink)
    ln -s "$(readlink -f "${SRC_HF}")" "${DST}"
    ;;
  *)
    echo "[error] COPY_MODE must be copy or symlink, got ${COPY_MODE}" >&2
    exit 1
    ;;
esac

# --- post-fix: drop `extra_special_tokens` list field from tokenizer_config.json ---
# Newer transformers write this as a LIST, which older eval envs (vLLM /
# older tokenizers) fail to load. The special tokens are already defined in
# tokenizer.json, so we simply remove the redundant key.
REPO_ROOT_ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX_SCRIPT="${REPO_ROOT_ABS}/scripts/fix_tokenizer_config.py"
PYTHON_BIN_FIX="${PYTHON_BIN_FIX:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"
if [[ -x "${PYTHON_BIN_FIX}" && -f "${FIX_SCRIPT}" ]]; then
  if [[ "${COPY_MODE}" == "copy" ]]; then
    TARGET_CFG="${DST}/tokenizer_config.json"
  else
    # For symlink mode, DST resolves to SRC_HF; fixing would mutate the source.
    # We still fix, because the file was presumably already written by training
    # and we want it usable everywhere. Use the resolved real file.
    TARGET_CFG="$(readlink -f "${DST}")/tokenizer_config.json"
  fi
  if [[ -f "${TARGET_CFG}" ]]; then
    "${PYTHON_BIN_FIX}" "${FIX_SCRIPT}" "${TARGET_CFG}" || \
      echo "[stage] WARN: tokenizer_config fix failed for ${TARGET_CFG}"
  fi
fi

cat > "${DST_ROOT}/${EXP_NAME}/eval_config_snippet.sh" <<EOF
# Verified working on the platform companion pretrain-eval pipeline.
BEGIN_STEP=0
END_STEP=99999999
EVAL_ITER=${STEP}
MAX_EVAL=1
EXP_NAME_BASE="${EXP_NAME}"
CKPT_BASE_PATH="$(readlink -f "${DST_ROOT}/${EXP_NAME}")"
EVAL_RESULT_PATH="$(readlink -f "${DST_ROOT}/${EXP_NAME}")/eval_result"
TASK_SET="all"
BACKEND="vllm"
TP_SIZE=1
API_WORKER_NUM=8
GPU_PER_NODE=8
EOF

echo "[done] staged: ${DST}"
echo "[done] config: ${DST_ROOT}/${EXP_NAME}/eval_config_snippet.sh"
