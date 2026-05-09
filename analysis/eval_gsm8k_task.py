"""
Task-level GSM8K evaluation on a trained checkpoint.

Reads `data/dev/math/samples.jsonl` (the same 256 GSM8K test items used as
the dev set during training), strips the gold answer, asks the model to
generate, then extracts the final numeric answer and compares to gold.

Reports exact-match accuracy on the numeric answer (the standard GSM8K
metric).

Usage:
  python analysis/eval_gsm8k_task.py \
      --ckpt outputs/.../final \
      --samples data/dev/math/samples.jsonl \
      --max-new-tokens 256 \
      --out-json eval_gsm8k_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


GOLD_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="HF model path or local final/ dir")
    p.add_argument("--samples", default="data/dev/math/samples.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1024)
    p.add_argument("--n-shot", type=int, default=4, help="few-shot examples")
    p.add_argument("--limit", type=int, default=-1, help="cap test items, -1 = all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-json", default="")
    p.add_argument("--tag", default="model")
    return p.parse_args()


def load_samples(path: Path) -> List[Tuple[str, str, str]]:
    """Return list of (question, gold_text, gold_number)."""
    items: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text"]
            # Format from analysis/export_real_dev_sets.py:
            #   "You are solving a math word problem.\n\nQuestion:\n<Q>\n\nAnswer:\n<A>"
            # Split on the first "Answer:\n" to recover Q and A.
            head, sep, ans_block = text.partition("Answer:\n")
            if not sep:
                continue
            q_marker = "Question:\n"
            qpos = head.find(q_marker)
            if qpos < 0:
                continue
            question = head[qpos + len(q_marker):].rstrip("\n")
            gold_match = GOLD_ANSWER_RE.search(ans_block)
            if not gold_match:
                continue
            gold_num = gold_match.group(1).replace(",", "")
            items.append((question.strip(), ans_block.strip(), gold_num))
    return items


# 4 fixed in-context exemplars (taken from publicly known GSM8K-style format).
# Kept short so input length stays modest.
FEWSHOT_EXAMPLES = [
    {
        "q": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "a": "Natalia sold 48/2 = 24 clips in May. In total, 48 + 24 = 72.\n#### 72",
    },
    {
        "q": "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much did she earn?",
        "a": "50 minutes is 50/60 = 5/6 hour. She earned 12 * 5/6 = 10.\n#### 10",
    },
    {
        "q": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money. Her parents decided to give her $15, and her grandparents twice as much as her parents. How much more money does Betty need?",
        "a": "Half of 100 is 50. Grandparents gave 2 * 15 = 30. Total = 50 + 15 + 30 = 95. She needs 100 - 95 = 5.\n#### 5",
    },
    {
        "q": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "a": "He writes 3 * 2 = 6 pages per friend per session. 2 friends -> 12 pages per session. 2 sessions/week -> 24 pages/week. 52 weeks -> 24 * 52 = 1248.\n#### 1248",
    },
]


def build_prompt(question: str, n_shot: int) -> str:
    parts: List[str] = []
    parts.append(
        "You are a careful math problem solver. For each problem, think step by step "
        "and then write the final numeric answer on a new line in the form '#### <number>'.\n"
    )
    for ex in FEWSHOT_EXAMPLES[:n_shot]:
        parts.append(f"Question:\n{ex['q']}\n\nAnswer:\n{ex['a']}\n")
    parts.append(f"Question:\n{question}\n\nAnswer:\n")
    return "\n".join(parts)


def extract_pred_number(generated_text: str) -> Optional[str]:
    """
    Extract the predicted numeric answer.

    Priority:
      1. The number that follows the LAST '####' marker, if present.
      2. Otherwise the LAST standalone number in the text.
    """
    # Cut off at the next "Question:" marker if the model started
    # producing a new few-shot example.
    next_q = generated_text.find("Question:")
    if next_q >= 0:
        generated_text = generated_text[:next_q]

    matches = list(re.finditer(r"####\s*(-?[\d,]+(?:\.\d+)?)", generated_text))
    if matches:
        return matches[-1].group(1).replace(",", "")
    nums = NUM_RE.findall(generated_text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def normalize_num(s: str) -> Optional[float]:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        return float(s)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    items = load_samples(samples_path)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[eval] loaded {len(items)} GSM8K items from {samples_path}", flush=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = torch.device(args.device)
    print(f"[eval] loading model from {args.ckpt} (dtype={args.dtype}, device={device})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    correct = 0
    total = 0
    bad_input = 0
    eg_records: List[Dict] = []
    t0 = time.time()
    with torch.no_grad():
        for idx, (q, gold_text, gold_num) in enumerate(items):
            prompt = build_prompt(q, args.n_shot)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
            if enc["input_ids"].shape[1] >= args.max_input_tokens:
                bad_input += 1
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_only = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_pred_number(gen_only)
            gold_f = normalize_num(gold_num)
            pred_f = normalize_num(pred) if pred is not None else None
            ok = (gold_f is not None and pred_f is not None and abs(gold_f - pred_f) < 1e-4)
            correct += int(ok)
            total += 1
            if idx < 5 or (idx + 1) % 50 == 0:
                eg_records.append({
                    "idx": idx,
                    "q": q[:200],
                    "gold": gold_num,
                    "pred": pred,
                    "ok": bool(ok),
                    "gen_head": gen_only[:240],
                })
                elapsed = time.time() - t0
                print(
                    f"[{idx+1:>4}/{len(items)}] acc={correct}/{total}={correct/total:.3f}  "
                    f"gold={gold_num} pred={pred} ok={ok}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

    acc = correct / max(total, 1)
    elapsed = time.time() - t0
    summary = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "samples_path": str(samples_path),
        "n_total": total,
        "n_correct": correct,
        "accuracy": acc,
        "n_input_truncated": bad_input,
        "elapsed_seconds": elapsed,
        "n_shot": args.n_shot,
        "max_new_tokens": args.max_new_tokens,
        "examples": eg_records,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
