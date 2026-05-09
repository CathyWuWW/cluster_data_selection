#!/usr/bin/env bash
# Launcher for Exp-mc (math+code only, reading dropped) verification:
#   m=24 meta-clusters, 600 steps
#   3 configs: hidden_base | hidden_utility_meta | hidden_random_meta
#   3 seeds  : 42 / 43 / 44
#   => 9 runs total, scheduled across 4 GPUs.
#
# Usage:
#   nohup bash scripts/run_exp_mc_launch.sh > outputs/exp_mc_launch.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"

# --- inputs ---
UTILITY_CSV_MC="${UTILITY_CSV_MC:-outputs/utility_mc_v1/utility_matrix_mc_sum_delta.csv}"
BASE_CLUSTER_IDS="${BASE_CLUSTER_IDS:-outputs/base_hidden_l2rm1pc_real/cluster_ids_initial.npy}"
N_META_CLUSTERS="${N_META_CLUSTERS:-24}"
TOTAL_ITERS="${TOTAL_ITERS:-600}"
SEEDS="${SEEDS:-42 43 44}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="outputs/exp_mc_m${N_META_CLUSTERS}_t${TOTAL_ITERS}_${TAG}"
GPUS=(0 1 2 3)

if [[ ! -f "${UTILITY_CSV_MC}" ]]; then
  echo "[error] missing ${UTILITY_CSV_MC} — run analysis/build_utility_mc.py first."
  exit 1
fi
if [[ ! -f "${BASE_CLUSTER_IDS}" ]]; then
  echo "[error] missing ${BASE_CLUSTER_IDS}"
  exit 1
fi

mkdir -p "${OUT_ROOT}/meta_clusters" "${OUT_ROOT}/runs" "${OUT_ROOT}/logs"
printf '%s\n' "${OUT_ROOT}" > outputs/latest_exp_mc_out_root.txt

cat > "${OUT_ROOT}/launch_manifest.json" <<EOF
{
  "tag": "${TAG}",
  "utility_csv_mc": "${UTILITY_CSV_MC}",
  "base_cluster_ids": "${BASE_CLUSTER_IDS}",
  "n_meta_clusters": ${N_META_CLUSTERS},
  "total_iters": ${TOTAL_ITERS},
  "seeds": "${SEEDS}",
  "dev_domains": ["math", "code"],
  "configs": ["hidden_base", "hidden_utility_meta", "hidden_random_meta"],
  "gpus": "${GPUS[*]}"
}
EOF

# --- step 1: build meta_clusters (utility + random) on the math+code utility ---
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

# --- step 2: enumerate 9 jobs (config x seed) ---
declare -a JOBS=()  # entries: "config|precomputed_ids|seed"
for seed in ${SEEDS}; do
  JOBS+=("hidden_base|${BASE_CLUSTER_IDS}|${seed}")
  JOBS+=("hidden_utility_meta|${UTIL_IDS}|${seed}")
  JOBS+=("hidden_random_meta|${RAND_IDS}|${seed}")
done

echo "[launch] total ${#JOBS[@]} jobs across ${#GPUS[@]} GPUs"

run_job() {
  local idx="$1"
  local entry="$2"
  local gpu="$3"
  IFS='|' read -r config precomp seed <<< "${entry}"
  local out_dir="${OUT_ROOT}/runs/${config}_s${seed}"
  local log="${OUT_ROOT}/logs/job_${idx}_${config}_s${seed}_gpu${gpu}.log"

  if [[ -e "${out_dir}/done.flag" ]]; then
    echo "[skip] ${out_dir} already done"
    return
  fi

  echo "[start] job=${idx} cfg=${config} seed=${seed} gpu=${gpu} -> ${out_dir}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    RUN_NAME="${config}" \
    PRECOMPUTED_IDS="${precomp}" \
    OUT_DIR="${out_dir}" \
    SEED="${seed}" \
    TOTAL_ITERS="${TOTAL_ITERS}" \
    MASTER_PORT="$((29541 + idx))" \
    bash scripts/run_exp_mc_one.sh \
      && touch "${out_dir}/done.flag"
  ) >"${log}" 2>&1
  echo "[end  ] job=${idx} cfg=${config} seed=${seed} gpu=${gpu} (rc=$?)"
}

# Schedule with at most ${#GPUS[@]} concurrent jobs.
N_GPU=${#GPUS[@]}
declare -A GPU_PID  # gpu_idx -> pid

wait_any_slot() {
  while true; do
    for i in $(seq 0 $((N_GPU - 1))); do
      local pid="${GPU_PID[$i]:-}"
      if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        echo "${i}"
        return
      fi
    done
    sleep 5
  done
}

idx=0
for entry in "${JOBS[@]}"; do
  slot=$(wait_any_slot)
  gpu="${GPUS[$slot]}"
  run_job "${idx}" "${entry}" "${gpu}" &
  GPU_PID[$slot]=$!
  idx=$((idx + 1))
  sleep 2
done

# wait for all to finish
for i in $(seq 0 $((N_GPU - 1))); do
  pid="${GPU_PID[$i]:-}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done

echo "[done] all ${#JOBS[@]} jobs finished. OUT_ROOT=${OUT_ROOT}"

# --- chain into evaluation if requested (default: yes) ---
if [[ "${AUTO_EVAL:-1}" == "1" ]]; then
  echo "[eval] auto-eval enabled, launching evaluation pipeline..."
  OUT_ROOT="${OUT_ROOT}" bash scripts/eval_exp_mc_all.sh \
    > "${OUT_ROOT}/eval_launch.log" 2>&1 || {
      echo "[eval] non-zero rc but continuing"
  }
  echo "[eval] evaluation done. Running summary..."
  "${PYTHON_BIN}" scripts/summarize_exp_mc.py \
    > "${OUT_ROOT}/summary_stdout.log" 2>&1 || true
  echo "[eval] summary written to ${OUT_ROOT}/summary.md"
fi

