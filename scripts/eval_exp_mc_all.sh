#!/usr/bin/env bash
# Evaluate every finished run under outputs/<OUT_ROOT>/runs/*/final
# on three metrics:
#   - LM dev loss (math + code) via a small Python helper
#   - GSM8K task accuracy (analysis/eval_gsm8k_task.py)
#   - MBPP pass@1            (analysis/eval_mbpp_task.py)
#
# Scheduled across 4 GPUs in parallel.
#
# Usage:
#   nohup bash scripts/eval_exp_mc_all.sh > outputs/exp_mc_eval.log 2>&1 &
#
# Optional env:
#   OUT_ROOT  : override (default = read from outputs/latest_exp_mc_out_root.txt)
#   LIMIT     : limit n samples per eval (debug)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/jizhicfs/wuyanning/miniconda3/envs/data_selection/bin/python}"
OUT_ROOT="${OUT_ROOT:-$(cat outputs/latest_exp_mc_out_root.txt)}"
LIMIT="${LIMIT:--1}"
GPUS=(0 1 2 3)

if [[ ! -d "${OUT_ROOT}/runs" ]]; then
  echo "[err] no runs dir under ${OUT_ROOT}"
  exit 1
fi

EVAL_DIR="${OUT_ROOT}/eval"
mkdir -p "${EVAL_DIR}/lm" "${EVAL_DIR}/gsm8k" "${EVAL_DIR}/mbpp" "${EVAL_DIR}/humaneval" "${EVAL_DIR}/logs"

# Discover all run dirs that have a `final/` checkpoint.
RUN_DIRS=()
for rd in "${OUT_ROOT}/runs"/*/; do
  rd="${rd%/}"
  if [[ -d "${rd}/final" ]]; then
    RUN_DIRS+=("${rd}")
  fi
done

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[err] no run has final/ ckpt under ${OUT_ROOT}/runs"
  exit 1
fi

echo "[eval] found ${#RUN_DIRS[@]} run dirs with final/"

# ---- helper: LM dev loss on math + code ----
eval_lm_one() {
  local ckpt="$1"
  local out_json="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" - <<PY
import json, math, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "${ckpt}"
out_json = "${out_json}"
device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(ckpt)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
model.eval()

def loss_on(jsonl_path: str, max_len: int = 256, batch_size: int = 4) -> float:
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
    total_tok, total_loss = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            n_tok = (labels[..., 1:] != -100).sum().item()
            total_loss += float(out.loss.item()) * n_tok
            total_tok += n_tok
    return total_loss / max(total_tok, 1)

results = {}
for d in ["math", "code"]:
    p = f"data/dev/{d}/samples.jsonl"
    if not Path(p).exists():
        continue
    l = loss_on(p)
    results[d] = l
    results[f"{d}_ppl"] = math.exp(l) if l < 100 else float("inf")
results["weighted_mc"] = (results.get("math", 0.0) + results.get("code", 0.0)) / 2.0
results["weighted_mc_ppl"] = math.exp(results["weighted_mc"]) if results["weighted_mc"] < 100 else float("inf")
results["ckpt"] = ckpt
Path(out_json).write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
PY
}

# ---- job queue ----
declare -a JOBS=()  # entries: "kind|ckpt|out_json|run_name"
for rd in "${RUN_DIRS[@]}"; do
  name="$(basename "${rd}")"
  ckpt="${rd}/final"
  JOBS+=("lm|${ckpt}|${EVAL_DIR}/lm/${name}.json|${name}")
  JOBS+=("gsm8k|${ckpt}|${EVAL_DIR}/gsm8k/${name}.json|${name}")
  JOBS+=("mbpp|${ckpt}|${EVAL_DIR}/mbpp/${name}.json|${name}")
  JOBS+=("humaneval|${ckpt}|${EVAL_DIR}/humaneval/${name}.json|${name}")
done

echo "[eval] scheduling ${#JOBS[@]} jobs across ${#GPUS[@]} GPUs"

run_eval_job() {
  local idx="$1" entry="$2" gpu="$3"
  IFS='|' read -r kind ckpt out_json name <<< "${entry}"
  local log="${EVAL_DIR}/logs/eval_${idx}_${kind}_${name}_gpu${gpu}.log"
  if [[ -e "${out_json}" && "${REUSE:-1}" == "1" ]]; then
    echo "[skip] ${out_json} exists"
    return
  fi
  echo "[start] eval idx=${idx} kind=${kind} name=${name} gpu=${gpu}"
  case "${kind}" in
    lm)
      eval_lm_one "${ckpt}" "${out_json}" "${gpu}" >"${log}" 2>&1
      ;;
    gsm8k)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" analysis/eval_gsm8k_task.py \
        --ckpt "${ckpt}" --tag "${name}" --out-json "${out_json}" \
        --limit "${LIMIT}" >"${log}" 2>&1
      ;;
    mbpp)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" analysis/eval_mbpp_task.py \
        --ckpt "${ckpt}" --tag "${name}" --out-json "${out_json}" \
        --limit "${LIMIT}" >"${log}" 2>&1
      ;;
    humaneval)
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" analysis/eval_humaneval_task.py \
        --ckpt "${ckpt}" --tag "${name}" --out-json "${out_json}" \
        --limit "${LIMIT}" >"${log}" 2>&1
      ;;
  esac
  echo "[end  ] eval idx=${idx} kind=${kind} name=${name} (rc=$?)"
}

N_GPU=${#GPUS[@]}
declare -A GPU_PID
wait_any_slot() {
  while true; do
    for i in $(seq 0 $((N_GPU - 1))); do
      local pid="${GPU_PID[$i]:-}"
      if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        echo "${i}"; return
      fi
    done
    sleep 5
  done
}

idx=0
for entry in "${JOBS[@]}"; do
  slot=$(wait_any_slot)
  gpu="${GPUS[$slot]}"
  run_eval_job "${idx}" "${entry}" "${gpu}" &
  GPU_PID[$slot]=$!
  idx=$((idx + 1))
  sleep 1
done

for i in $(seq 0 $((N_GPU - 1))); do
  pid="${GPU_PID[$i]:-}"
  [[ -n "${pid}" ]] && wait "${pid}" || true
done

echo "[done] all eval jobs finished. results under ${EVAL_DIR}"
