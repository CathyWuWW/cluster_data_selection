"""
Fix HF tokenizer_config.json files written by newer transformers/tokenizers
that put `extra_special_tokens` as a LIST, which breaks older evaluation
environments (vLLM / older tokenizers) that expect a DICT.

The special tokens themselves are already fully defined in tokenizer.json,
so the cleanest fix is to simply REMOVE the `extra_special_tokens` key.

Usage:
    python scripts/fix_tokenizer_config.py <path> [<path> ...]
    python scripts/fix_tokenizer_config.py --glob 'outputs/**/tokenizer_config.json'

For each file:
  - If `extra_special_tokens` is not present or is already a dict: skip.
  - Else: write a `.bak` snapshot (only if no .bak exists yet) and remove the key.

Safe to re-run.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="tokenizer_config.json files to fix")
    p.add_argument("--glob", default="", help="glob pattern; use with recursive **/ ")
    p.add_argument("--dry-run", action="store_true", help="only report what would change")
    return p.parse_args()


def collect_targets(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for s in args.paths:
        p = Path(s)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(Path(s).rglob("tokenizer_config.json"))
    if args.glob:
        for m in _glob.glob(args.glob, recursive=True):
            files.append(Path(m))
    # dedupe, stable order
    seen, uniq = set(), []
    for f in files:
        r = f.resolve()
        if r in seen:
            continue
        seen.add(r)
        uniq.append(f)
    return uniq


def fix_one(path: Path, dry: bool) -> tuple[str, str]:
    """Return (status, detail). status in {fixed, skip, error}."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as e:
        return "error", f"cannot parse: {e}"

    v = data.get("extra_special_tokens", "__MISSING__")
    if v == "__MISSING__":
        return "skip", "no extra_special_tokens field"
    if isinstance(v, dict):
        return "skip", "already a dict"
    if not isinstance(v, list):
        return "skip", f"unexpected type: {type(v).__name__}"

    if dry:
        return "fixed", f"(dry-run) would drop list of {len(v)} tokens"

    # backup only if no .bak exists
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)

    del data["extra_special_tokens"]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return "fixed", f"dropped list of {len(v)} tokens (backup: {bak.name})"


def main() -> None:
    args = parse_args()
    files = collect_targets(args)
    if not files:
        print("[fix] no target files found")
        sys.exit(0)

    n_fixed = n_skip = n_err = 0
    for f in files:
        st, detail = fix_one(f, args.dry_run)
        tag = {"fixed": "[fix]", "skip": "[skip]", "error": "[err ]"}[st]
        print(f"{tag} {f}: {detail}")
        if st == "fixed":
            n_fixed += 1
        elif st == "skip":
            n_skip += 1
        else:
            n_err += 1

    print(f"\n[summary] fixed={n_fixed} skip={n_skip} error={n_err} total={len(files)}")


if __name__ == "__main__":
    main()
