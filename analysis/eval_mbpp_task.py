"""
Task-level MBPP-style code evaluation on a trained checkpoint.

Reads `data/dev/code/samples.jsonl` (the same 256 MBPP test items used as
the dev set during training). Each record is:

  You are solving a coding task.

  Problem:
  <natural language problem>

  Tests:
  assert ...
  assert ...

  Reference solution:
  <python source>

We strip the reference solution, prompt the model with
problem + tests, take the first ```python code block from the generation,
exec it in a sandbox, and run the asserts. A sample is "pass" iff every
assert returns without exception.

Reports pass@1 (greedy decode, 1 sample per item).

Usage:
  python analysis/eval_mbpp_task.py \\
      --ckpt outputs/.../final \\
      --samples data/dev/code/samples.jsonl \\
      --max-new-tokens 384 \\
      --out-json eval_mbpp_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--samples", default="data/dev/code/samples.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--max-input-tokens", type=int, default=1536)
    p.add_argument("--n-shot", type=int, default=2)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-json", default="")
    p.add_argument("--tag", default="model")
    p.add_argument("--exec-timeout", type=float, default=8.0)
    return p.parse_args()


def parse_record(rec: Dict) -> Optional[Tuple[str, List[str]]]:
    """Return (prompt_problem_with_tests, list_of_assert_lines) or None."""
    text = rec["text"]
    # Locate the three required markers.
    p_marker = "Problem:\n"
    t_marker = "\n\nTests:\n"
    r_marker = "\n\nReference solution:\n"
    p_pos = text.find(p_marker)
    t_pos = text.find(t_marker)
    r_pos = text.find(r_marker)
    if p_pos < 0 or t_pos < 0 or r_pos < 0 or not (p_pos < t_pos < r_pos):
        return None
    problem = text[p_pos + len(p_marker):t_pos].strip()
    tests_block = text[t_pos + len(t_marker):r_pos].strip()
    asserts = [
        line.rstrip()
        for line in tests_block.splitlines()
        if line.strip().startswith("assert")
    ]
    if not asserts:
        return None
    return problem, asserts


def load_samples(path: Path) -> List[Tuple[str, List[str]]]:
    items: List[Tuple[str, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parsed = parse_record(rec)
            if parsed is not None:
                items.append(parsed)
    return items


# Two short fixed in-context examples (style-matched to MBPP).
FEWSHOT_EXAMPLES = [
    {
        "problem": "Write a python function to find the sum of the first n natural numbers.",
        "tests": [
            "assert sum_natural(5) == 15",
            "assert sum_natural(0) == 0",
            "assert sum_natural(10) == 55",
        ],
        "code": (
            "def sum_natural(n):\n"
            "    return n * (n + 1) // 2\n"
        ),
    },
    {
        "problem": "Write a function to check whether the given string is a palindrome.",
        "tests": [
            'assert is_palindrome("racecar") == True',
            'assert is_palindrome("hello") == False',
            'assert is_palindrome("a") == True',
        ],
        "code": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
        ),
    },
]


def build_prompt(problem: str, asserts: List[str], n_shot: int) -> str:
    parts: List[str] = []
    parts.append(
        "You are an expert Python programmer. For each task, write a single "
        "self-contained Python solution that makes the provided assert "
        "statements all pass. Wrap the final code in a ```python ... ``` block.\n"
    )
    for ex in FEWSHOT_EXAMPLES[:n_shot]:
        tests = "\n".join(ex["tests"])
        parts.append(
            f"Problem:\n{ex['problem']}\n\nTests:\n{tests}\n\n"
            f"Solution:\n```python\n{ex['code']}```\n"
        )
    tests = "\n".join(asserts)
    parts.append(
        f"Problem:\n{problem}\n\nTests:\n{tests}\n\nSolution:\n```python\n"
    )
    return "\n".join(parts)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(generated_text: str) -> str:
    """Pull the first python code block out of the generation.

    The prompt ends with '```python\\n', so the model's output starts inside
    a code block. We close at the first triple-backtick.
    """
    # Prepend back the open fence so the regex sees a complete block.
    wrapped = "```python\n" + generated_text
    m = CODE_BLOCK_RE.search(wrapped)
    if m:
        return m.group(1)
    # Fallback: cut off at obvious stop tokens.
    for stop in ["\nProblem:", "\n```", "\nSolution:", "\nTests:"]:
        idx = generated_text.find(stop)
        if idx >= 0:
            generated_text = generated_text[:idx]
            break
    return generated_text


def _exec_worker(code_and_tests: str, return_dict) -> None:
    """Run inside a subprocess. Sets return_dict['ok']=True on full pass."""
    try:
        ns: Dict[str, object] = {"__name__": "__main__"}
        exec(compile(code_and_tests, "<mbpp_eval>", "exec"), ns, ns)
        return_dict["ok"] = True
    except BaseException as e:  # noqa: BLE001
        return_dict["ok"] = False
        return_dict["err"] = f"{type(e).__name__}: {e}"


def run_tests_subprocess(code: str, asserts: List[str], timeout: float) -> Tuple[bool, str]:
    """Exec code + asserts in a subprocess with a wall-clock timeout."""
    full = code.rstrip() + "\n\n" + "\n".join(asserts) + "\n"
    manager = mp.Manager()
    rd = manager.dict()
    proc = mp.Process(target=_exec_worker, args=(full, rd))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, "timeout"
    ok = bool(rd.get("ok", False))
    err = str(rd.get("err", "") or "")
    return ok, err


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    items = load_samples(samples_path)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"[eval] loaded {len(items)} MBPP items from {samples_path}", flush=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
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
        for idx, (problem, asserts) in enumerate(items):
            prompt = build_prompt(problem, asserts, args.n_shot)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=args.max_input_tokens)
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
            code = extract_code(gen_only)
            ok, err = run_tests_subprocess(code, asserts, args.exec_timeout)
            correct += int(ok)
            total += 1
            if idx < 3 or (idx + 1) % 50 == 0:
                eg_records.append({
                    "idx": idx,
                    "problem": problem[:200],
                    "asserts_head": asserts[0] if asserts else "",
                    "ok": bool(ok),
                    "err": err[:200],
                    "code_head": code[:200],
                })
                elapsed = time.time() - t0
                print(
                    f"[{idx+1:>4}/{len(items)}] pass={correct}/{total}={correct/total:.3f}  "
                    f"ok={ok}  err={err[:80]}  elapsed={elapsed:.1f}s",
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
        "pass_at_1": acc,
        "n_input_truncated": bad_input,
        "elapsed_seconds": elapsed,
        "n_shot": args.n_shot,
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
