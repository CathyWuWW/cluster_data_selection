from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from datasets import load_dataset

DatasetSpec = Tuple[str, str | None, str]

DEFAULT_SPECS: Dict[str, DatasetSpec] = {
    "math": ("openai/gsm8k", "main", "test"),
    "reading": ("squad", None, "validation"),
    "code": ("mbpp", None, "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real public dev sets to JSONL files with a text field."
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--samples-per-ability", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def shuffled_take(records: Iterable[dict], count: int, seed: int) -> List[dict]:
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < count:
        raise ValueError(f"Only {len(items)} records available, need at least {count}")
    return items[:count]


def format_math(record: dict) -> str:
    return (
        "You are solving a math word problem.\n\n"
        f"Question:\n{record['question'].strip()}\n\n"
        f"Answer:\n{record['answer'].strip()}"
    )


def format_reading(record: dict) -> str:
    answers = [a.strip() for a in record.get("answers", {}).get("text", []) if a and a.strip()]
    answer_text = " | ".join(answers) if answers else "No answer provided."
    return (
        "You are answering a reading comprehension question.\n\n"
        f"Title: {str(record.get('title', '')).strip()}\n\n"
        f"Context:\n{record['context'].strip()}\n\n"
        f"Question:\n{record['question'].strip()}\n\n"
        f"Answer:\n{answer_text}"
    )


def format_code(record: dict) -> str:
    setup = str(record.get("test_setup_code", "")).strip()
    tests = [str(x).strip() for x in record.get("test_list", []) if str(x).strip()]
    challenge_tests = [str(x).strip() for x in record.get("challenge_test_list", []) if str(x).strip()]
    parts = [
        "You are solving a coding task.",
        "",
        "Problem:",
        record["text"].strip(),
    ]
    if setup:
        parts.extend(["", "Test setup:", setup])
    if tests:
        parts.extend(["", "Tests:", "\n".join(tests)])
    if challenge_tests:
        parts.extend(["", "Challenge tests:", "\n".join(challenge_tests)])
    parts.extend(["", "Reference solution:", record["code"].strip()])
    return "\n".join(parts)


FORMATTERS: Dict[str, Callable[[dict], str]] = {
    "math": format_math,
    "reading": format_reading,
    "code": format_code,
}


def load_records(spec: DatasetSpec) -> List[dict]:
    name, config, split = spec
    dataset = load_dataset(name, config, split=split)
    return list(dataset)


def export_ability(
    ability: str,
    out_dir: Path,
    spec: DatasetSpec,
    formatter: Callable[[dict], str],
    samples_per_ability: int,
    seed: int,
) -> dict:
    records = load_records(spec)
    selected = shuffled_take(records, samples_per_ability, seed)
    out_path = out_dir / "samples.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps({"text": formatter(record)}, ensure_ascii=False) + "\n")
    return {
        "dataset": spec[0],
        "config": spec[1],
        "split": spec[2],
        "count": len(selected),
        "output_file": str(out_path),
    }


def main() -> None:
    args = parse_args()
    out_root: Path = args.out_root
    manifest: dict = {
        "samples_per_ability": args.samples_per_ability,
        "seed": args.seed,
        "abilities": {},
    }

    for ability, spec in DEFAULT_SPECS.items():
        out_dir = out_root / ability
        ensure_empty_dir(out_dir, overwrite=args.overwrite)
        manifest["abilities"][ability] = export_ability(
            ability=ability,
            out_dir=out_dir,
            spec=spec,
            formatter=FORMATTERS[ability],
            samples_per_ability=args.samples_per_ability,
            seed=args.seed,
        )

    manifest_path = out_root / "real_dev_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] Exported real dev sets to {out_root}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
