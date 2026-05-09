#!/usr/bin/env bash
# One-shot smoke test for the Joint Loss Level 1 pipeline.
# NOT a permanent script — meant to be invoked by hand once after wiring.
#
# Verifies on a single GPU + ~4 steps that:
#   - prototype init runs and writes prototype_init.npy + meta.json
#   - forward/backward/step all succeed under joint_loss.enabled=true
#   - prototype_history.jsonl gets the init row + at least one step row
#   - the run finishes and writes prototype_final.npy
#
# Output goes to outputs/_smoke_joint_l1_<TAG>/, which can be removed
# afterwards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"
GPU="${GPU:-0}"
TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/_smoke_joint_l1_${TAG}"
META_IDS="${META_IDS:-outputs/exp_mc_1p5b_pilot_m24_t1000_s42_20260506_162512/meta_clusters/utility_m24/meta_cluster_ids.npy}"

if [[ ! -f "${META_IDS}" ]]; then
  echo "[error] missing META_IDS=${META_IDS}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
RUN_CONFIG="${OUT_DIR}/run_config.yaml"

OUT_DIR_VALUE="${OUT_DIR}" \
RUN_CONFIG_PATH="${RUN_CONFIG}" \
PRECOMPUTED_IDS_PATH="${META_IDS}" \
BASE_CONFIG_PATH="configs/default.yaml" \
"${PYTHON_BIN}" - <<'PY'
import os
from omegaconf import OmegaConf
cfg = OmegaConf.load(os.environ["BASE_CONFIG_PATH"])

# Minimal but realistic 1.5B path
cfg.model.path = "Qwen/Qwen2.5-1.5B"
cfg.model.attn_impl = "sdpa"
cfg.model.max_length = 512
cfg.model.gradient_checkpointing = False

cfg.data.train_dir = "local_data/slimpajama_6b_balanced/train"
cfg.data.dev_dir = "data/dev/math"
cfg.data.dev_domains = [
    {"name": "math", "dir": "data/dev/math", "weight": 1.0},
]
cfg.data.text_field = "text"
cfg.data.dev_num = 8
cfg.data.dev_seed = 42
cfg.data.eval_format = "text"
cfg.data.num_workers = 0

cfg.training.total_iters = 4
cfg.training.batch_size = 1
cfg.training.gradient_accumulation_steps = 1
cfg.training.eval_batch_size = 2
cfg.training.eval_interval = 999999  # skip mid-training eval
cfg.training.no_eval_at_start = True
cfg.training.save_interval = 999999
cfg.training.save_dir = os.environ["OUT_DIR_VALUE"]
cfg.training.lr = 1e-5
cfg.training.warmup_iters = 2
cfg.training.seed = 42
cfg.training.log_interval = 1

cfg.clustering.precomputed_ids_path = os.environ["PRECOMPUTED_IDS_PATH"]
cfg.clustering.recluster_interval = -1

cfg.pmp.update_interval = 2
cfg.pmp.dev_batch_size = 2
cfg.pmp.drop_bad_clusters = False
if hasattr(cfg.pmp, "utility_prior"):
    cfg.pmp.utility_prior.enabled = False

# Joint loss: J0 (sanity, lam=0)
cfg.joint_loss.enabled = True
cfg.joint_loss.lam = 0.0
cfg.joint_loss.distance = "cosine"
cfg.joint_loss.layer_idx = 14
cfg.joint_loss.proto_lr_multiplier = 10.0
cfg.joint_loss.init.source = "feature_mean"
cfg.joint_loss.init.batch_size = 4
cfg.joint_loss.init.max_samples_per_cluster = 4   # only ~96 samples -> very fast
cfg.joint_loss.weight.mode = "uniform"
cfg.joint_loss.log_interval = 0  # piggyback on pmp.update_interval=2

cfg.deepspeed.enabled = False

OmegaConf.save(cfg, os.environ["RUN_CONFIG_PATH"])
print(f"[smoke] wrote config: {os.environ['RUN_CONFIG_PATH']}")
PY

echo "[smoke] launching on GPU=${GPU}, out=${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" \
HF_HOME="${REPO_ROOT}/.hf_cache" \
HF_DATASETS_CACHE="${REPO_ROOT}/.hf_cache/datasets" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON_BIN}" train.py --config "${RUN_CONFIG}" 2>&1 | tee "${OUT_DIR}/smoke.log"

echo
echo "[smoke] artifacts:"
ls -la "${OUT_DIR}/"
echo
echo "[smoke] prototype_history.jsonl content:"
cat "${OUT_DIR}/prototype_history.jsonl" 2>/dev/null || echo "(missing)"
