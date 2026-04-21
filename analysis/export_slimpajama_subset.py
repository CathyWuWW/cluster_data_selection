"""
Export reproducible SlimPajama-6B subsets for representation validation.

The training pipeline in this repository expects JSON/JSONL records with a
`text` field. This script writes that field plus lightweight metadata so the
same files can be used both for training and for representation diagnostics.

Example:
    python analysis/export_slimpajama_subset.py \
        --out-dir local_data/slimpajama_6b_balanced \
        --mode stratified \
        --train-size 100000 \
        --repr-size 10000 \
        --val-size 5000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_DATASET = "DKYoon/SlimPajama-6B"
DEFAULT_SOURCES = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SlimPajama-6B JSONL subsets for cluster validation."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="local_data/slimpajama_6b")
    parser.add_argument(
        "--mode",
        choices=["stratified", "natural"],
        default="stratified",
        help=(
            "stratified balances records across source domains; natural keeps "
            "the shuffled stream distribution."
        ),
    )
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--repr-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--source-field", default="meta.redpajama_set_name")
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source labels used by stratified mode.",
    )
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="If >0, truncate text to this many characters before writing.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="If >0, stop after scanning this many records even if quotas are not met.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=100_000,
        help="Streaming shuffle buffer size.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the dataset normally instead of using Hugging Face streaming.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def get_nested(record: Mapping[str, Any], dotted_path: str) -> Any:
    cur: Any = record
    for key in dotted_path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def clean_text(text: Any, min_chars: int, max_chars: int) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < min_chars:
        return None
    if max_chars > 0:
        text = text[:max_chars]
    return text


def split_quotas(total: int, labels: List[str]) -> Dict[str, int]:
    if total <= 0:
        return {label: 0 for label in labels}
    base = total // len(labels)
    remainder = total % len(labels)
    quotas = {label: base for label in labels}
    for label in labels[:remainder]:
        quotas[label] += 1
    return quotas


def open_writers(out_dir: Path):
    paths = {
        "train": out_dir / "train" / "train.jsonl",
        "repr": out_dir / "repr" / "repr_eval.jsonl",
        "val": out_dir / "valid" / "slimpajama_valid.jsonl",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    return paths, handles


def write_record(handle, text: str, source: str, record: Mapping[str, Any], sample_id: int):
    payload = {
        "text": text,
        "source": source,
        "sample_id": sample_id,
    }
    meta = record.get("meta")
    if isinstance(meta, Mapping):
        payload["meta"] = dict(meta)
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_dataset(args: argparse.Namespace) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it on the server with "
            "`pip install datasets pyarrow`."
        ) from exc

    streaming = not args.no_streaming
    ds = load_dataset(args.dataset, split=args.split, streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
        return ds

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    return (ds[i] for i in indices)


def export_natural(args: argparse.Namespace, rows: Iterable[Mapping[str, Any]], handles):
    targets = {
        "train": args.train_size,
        "repr": args.repr_size,
        "val": args.val_size,
    }
    counts = Counter()
    source_counts = {name: Counter() for name in targets}
    scanned = 0
    sample_id = 0

    for record in rows:
        scanned += 1
        if args.max_scan > 0 and scanned > args.max_scan:
            break

        text = clean_text(record.get(args.text_field), args.min_chars, args.max_chars)
        if text is None:
            continue

        split_name = next((name for name, target in targets.items() if counts[name] < target), None)
        if split_name is None:
            break

        source = get_nested(record, args.source_field) or "unknown"
        write_record(handles[split_name], text, str(source), record, sample_id)
        counts[split_name] += 1
        source_counts[split_name][str(source)] += 1
        sample_id += 1

    return counts, source_counts, scanned


def export_stratified(args: argparse.Namespace, rows: Iterable[Mapping[str, Any]], handles):
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    targets = {
        "train": split_quotas(args.train_size, sources),
        "repr": split_quotas(args.repr_size, sources),
        "val": split_quotas(args.val_size, sources),
    }
    counts = {name: Counter() for name in targets}
    source_counts = {name: Counter() for name in targets}
    scanned = 0
    sample_id = 0

    def all_done() -> bool:
        for split_name, quotas in targets.items():
            for source, quota in quotas.items():
                if counts[split_name][source] < quota:
                    return False
        return True

    for record in rows:
        scanned += 1
        if args.max_scan > 0 and scanned > args.max_scan:
            break

        source = get_nested(record, args.source_field) or "unknown"
        source = str(source)
        if source not in sources:
            continue

        text = clean_text(record.get(args.text_field), args.min_chars, args.max_chars)
        if text is None:
            continue

        split_name = None
        for candidate in ("train", "repr", "val"):
            if counts[candidate][source] < targets[candidate][source]:
                split_name = candidate
                break
        if split_name is None:
            if all_done():
                break
            continue

        write_record(handles[split_name], text, source, record, sample_id)
        counts[split_name][source] += 1
        source_counts[split_name][source] += 1
        sample_id += 1

        if all_done():
            break

    flat_counts = Counter({name: sum(c.values()) for name, c in counts.items()})
    return flat_counts, source_counts, scanned


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists and is not empty: {out_dir}. "
            "Pass --overwrite to replace files."
        )

    paths, handles = open_writers(out_dir)
    try:
        rows = iter_dataset(args)
        if args.mode == "natural":
            counts, source_counts, scanned = export_natural(args, rows, handles)
        else:
            counts, source_counts, scanned = export_stratified(args, rows, handles)
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "mode": args.mode,
        "seed": args.seed,
        "text_field": args.text_field,
        "source_field": args.source_field,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "scanned_records": scanned,
        "outputs": {name: str(path) for name, path in paths.items()},
        "counts": dict(counts),
        "source_counts": {
            name: dict(counter) for name, counter in source_counts.items()
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
