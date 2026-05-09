"""
HumanEval pass@1 evaluation on a trained checkpoint.

Reads `data/dev/humaneval/samples.jsonl` (164 OpenAI HumanEval items, exported
once from `datasets.load_dataset('openai_humaneval')` with the official fields:
task_id, prompt, canonical_solution, test, entry_point).

Procedure:
  1. Feed the model `prompt` (function signature + docstring) and let it
     greedy-decode a continuation up to --max-new-tokens.
  2. Cut the continuation at the first sign of a new top-level def/class or
     a triple-backtick fence.
  3. Concatenate prompt + cleaned_completion + "\n" + test + f"\ncheck({entry_point})\n",
     exec in a fresh subprocess with a wall-clock timeout.
  4. Pass iff exec returns without raising.

Reports pass@1 (greedy, n=1).

Usage:
  python analysis/eval_humaneval_task.py \\
      --ckpt outputs/.../final \\
      --samples data/dev/humaneval/samples.jsonl \\
      --max-new-tokens 384 \\
      --out-json eval_humaneval_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--samples", default="data/dev/humaneval/samples.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-input-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-json", default="")
    p.add_argument("--tag", default="model")
    p.add_argument("--exec-timeout", type=float, default=8.0)
    return p.parse_args()


def load_samples(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# Stop markers that indicate a new top-level definition started, which
# means the model is "done" with the requested function.
STOP_PATTERNS = [
    re.compile(r"\n\s*def\s+\w"),
    re.compile(r"\n\s*class\s+\w"),
    re.compile(r"\nif __name__"),
    re.compile(r"\nprint\("),
    re.compile(r"\n#\s*Test"),
    re.compile(r"\n```"),
]


def truncate_completion(completion: str) -> str:
    """Cut at the first new top-level def / class / fence / __main__."""
    cut = len(completion)
    for pat in STOP_PATTERNS:
        m = pat.search(completion)
        if m:
            cut = min(cut, m.start())
    return completion[:cut]


def _exec_worker(code: str, return_dict) -> None:
    try:
        ns: Dict[str, object] = {"__name__": "__main__"}
        exec(compile(code, "<humaneval>", "exec"), ns, ns)
        return_dict["ok"] = True
    except BaseException as e:  # noqa: BLE001
        return_dict["ok"] = False
        return_dict["err"] = f"{type(e).__name__}: {e}"


def run_subprocess(full_code: str, timeout: float) -> Tuple[bool, str]:
    manager = mp.Manager()
    rd = manager.dict()
    proc = mp.Process(target=_exec_worker, args=(full_code, rd))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, "timeout"
    return bool(rd.get("ok", False)), str(rd.get("err", "") or "")


def main() -> None:
    args = parse_args()
    items = load_samples(Path(args.samples))
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[eval] loaded {len(items)} HumanEval items", flush=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    device = torch.device(args.device)
    print(f"[eval] loading model from {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    correct = 0
    total = 0
    eg_records: List[Dict] = []
    t0 = time.time()
    with torch.no_grad():
        for idx, rec in enumerate(items):
            prompt = rec["prompt"]
            test = rec["test"]
            entry_point = rec["entry_point"]
            task_id = rec["task_id"]

            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=args.max_input_tokens)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            gen_only = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            completion = truncate_completion(gen_only)
            full = prompt + completion + "\n\n" + test + f"\ncheck({entry_point})\n"
            ok, err = run_subprocess(full, args.exec_timeout)
            correct += int(ok)
            total += 1
            if idx < 3 or (idx + 1) % 25 == 0:
                eg_records.append({
                    "idx": idx,
                    "task_id": task_id,
                    "ok": bool(ok),
                    "err": err[:200],
                    "completion_head": completion[:240],
                })
                elapsed = time.time() - t0
                print(
                    f"[{idx+1:>4}/{len(items)}] pass={correct}/{total}={correct/total:.3f}  "
                    f"task={task_id} ok={ok} err={err[:80]} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    acc = correct / max(total, 1)
    elapsed = time.time() - t0
    summary = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "samples_path": str(args.samples),
        "n_total": total,
        "n_correct": correct,
        "pass_at_1": acc,
        "elapsed_seconds": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "examples": eg_records,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
