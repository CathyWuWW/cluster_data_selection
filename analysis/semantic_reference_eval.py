"""
Semantic-reference evaluation for hidden representations.

This script treats E5/BGE-style sentence embedding models as semantic reference
spaces and checks whether hidden representations preserve their nearest-neighbor
sets and similarity rankings. It is not a ground-truth semantic benchmark; it is
a reference-space compatibility test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from nearest_neighbor_sanity import (
    RepresentationSpec,
    Sample,
    extract_hidden_features,
    get_num_layers,
    l2_normalize,
    load_model_and_tokenizer,
    load_samples,
    pool_hidden,
    remove_top_pcs,
    resolve_layer,
    source_counts,
    torch_dtype,
    write_samples,
)


@dataclass
class FeatureSpec:
    name: str
    family: str
    detail: str
    normalize: str = "l2"
    remove_top_pcs: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hidden representations against E5/BGE semantic reference spaces."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "random", "stratified"],
        default="stratified",
    )
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--candidate-count", type=int, default=500)
    parser.add_argument("--top-k", default="10,50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--hidden-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--hidden-tokenizer", default=None)
    parser.add_argument(
        "--hidden-representations",
        default=(
            "layer12_mean_l2:middle:mean:l2:0;"
            "layer12_mean_l2_rm1pc:middle:mean:l2:1;"
            "layer23_mean_l2:last:mean:l2:0;"
            "layer23_mean_l2_rm1pc:last:mean:l2:1"
        ),
        help=(
            "Semicolon-separated specs: name:layer:pooling:normalize:remove_top_pcs. "
            "Layer may be first/middle/last or an integer."
        ),
    )
    parser.add_argument(
        "--reference-models",
        default="intfloat/e5-small-v2,BAAI/bge-small-en-v1.5",
        help="Comma-separated sentence embedding models used as references.",
    )
    parser.add_argument(
        "--sentence-prefix",
        default="auto",
        help="Use auto for E5 'passage: ' prefix, or pass an explicit prefix.",
    )
    parser.add_argument("--include-tfidf", action="store_true")
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--random-dim", type=int, default=768)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-features", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    chars = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("-")
    result = "".join(chars).strip("-._")
    while "--" in result:
        result = result.replace("--", "-")
    return result or "run"


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_run_dir(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = Path(args.out_dir)
    if args.run_name:
        run_name = slug(args.run_name)
    else:
        sample_size = "all" if args.max_samples <= 0 else str(args.max_samples)
        run_name = "_".join(
            [
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                slug(args.hidden_model.split("/")[-1]),
                f"n{sample_size}",
                args.sample_strategy,
                f"q{args.query_count}",
                f"cand{args.candidate_count}",
                f"k-{slug(args.top_k)}",
                f"seed{args.seed}",
            ]
        )
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return root, run_dir


def parse_hidden_specs(raw: str, num_layers: int) -> List[RepresentationSpec]:
    specs: List[RepresentationSpec] = []
    for raw_spec in raw.split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        name, layer, pooling, normalize, remove_top_pcs = raw_spec.split(":")
        specs.append(
            RepresentationSpec(
                name=name,
                layer=resolve_layer(layer, num_layers),
                pooling=pooling,
                normalize=(normalize.lower() == "l2"),
                remove_top_pcs=int(remove_top_pcs),
            )
        )
    return specs


def transform_hidden_features(
    raw_features: Dict[str, np.ndarray],
    specs: List[RepresentationSpec],
) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    features: Dict[str, np.ndarray] = {}
    feature_specs: Dict[str, FeatureSpec] = {}
    for spec in specs:
        x = raw_features[spec.name].astype(np.float32)
        if spec.remove_top_pcs > 0:
            x = remove_top_pcs(x, spec.remove_top_pcs).astype(np.float32)
        if spec.normalize:
            x = l2_normalize(x).astype(np.float32)
        features[spec.name] = x
        feature_specs[spec.name] = FeatureSpec(
            name=spec.name,
            family="hidden",
            detail=f"layer={spec.layer},pooling={spec.pooling}",
            normalize="l2" if spec.normalize else "raw",
            remove_top_pcs=spec.remove_top_pcs,
        )
    return features, feature_specs


def hidden_features(args: argparse.Namespace, samples: List[Sample]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    model_args = argparse.Namespace(
        model=args.hidden_model,
        tokenizer=args.hidden_tokenizer,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        progress_every=args.progress_every,
    )
    model, tokenizer, device = load_model_and_tokenizer(model_args)
    num_layers = get_num_layers(model)
    specs = parse_hidden_specs(args.hidden_representations, num_layers)
    print(f"[hidden] model={args.hidden_model} specs={[s.name for s in specs]}", flush=True)
    raw = extract_hidden_features(model_args, samples, model, tokenizer, device, specs)
    return transform_hidden_features(raw, specs)


def sentence_prefix_for_model(model_name: str, prefix_arg: str) -> str:
    if prefix_arg == "auto":
        return "passage: " if "e5" in model_name.lower() else ""
    return prefix_arg


def extract_sentence_features(
    args: argparse.Namespace,
    model_name: str,
    samples: List[Sample],
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    prefix = sentence_prefix_for_model(model_name, args.sentence_prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    features: List[np.ndarray] = []
    total_batches = math.ceil(len(samples) / args.batch_size)
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(samples), args.batch_size), start=1):
            batch = samples[start : start + args.batch_size]
            texts = [prefix + s.text for s in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = pool_hidden(outputs.last_hidden_state, encoded["attention_mask"], "mean")
            features.append(pooled.cpu().float().numpy())
            if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                print(f"[sentence] model={model_name} batch={batch_idx}/{total_batches}", flush=True)
    return l2_normalize(np.concatenate(features, axis=0).astype(np.float32))


def reference_features(args: argparse.Namespace, samples: List[Sample]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    features: Dict[str, np.ndarray] = {}
    specs: Dict[str, FeatureSpec] = {}
    for model_name in parse_csv_list(args.reference_models):
        name = f"ref_{slug(model_name)}"
        print(f"[reference] model={model_name}", flush=True)
        features[name] = extract_sentence_features(args, model_name, samples)
        specs[name] = FeatureSpec(name=name, family="reference", detail=model_name)
    return features, specs


def baseline_features(args: argparse.Namespace, samples: List[Sample]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    features: Dict[str, np.ndarray] = {}
    specs: Dict[str, FeatureSpec] = {}
    rng = np.random.default_rng(args.seed)
    if args.include_random:
        name = f"random_normal_d{args.random_dim}"
        features[name] = l2_normalize(rng.normal(size=(len(samples), args.random_dim)).astype(np.float32))
        specs[name] = FeatureSpec(name=name, family="random", detail=f"dim={args.random_dim}")
    if args.include_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer

        name = f"tfidf_word_{args.tfidf_max_features}"
        vectorizer = TfidfVectorizer(
            max_features=args.tfidf_max_features,
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            dtype=np.float32,
        )
        features[name] = l2_normalize(vectorizer.fit_transform([s.text for s in samples]).toarray().astype(np.float32))
        specs[name] = FeatureSpec(name=name, family="lexical", detail=f"word_tfidf_max_features={args.tfidf_max_features}")
    return features, specs


def choose_query_indices(samples: List[Sample], query_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(query_count, len(samples))
    return np.sort(rng.choice(len(samples), size=count, replace=False))


def choose_candidate_indices(
    samples: List[Sample],
    query_idx: int,
    candidate_count: int,
    seed: int,
    cross_source: bool,
) -> np.ndarray:
    query_source = samples[query_idx].source
    candidates = [
        idx
        for idx, sample in enumerate(samples)
        if idx != query_idx and (not cross_source or sample.source != query_source)
    ]
    if candidate_count <= 0 or candidate_count >= len(candidates):
        return np.array(candidates, dtype=np.int64)
    rng = np.random.default_rng(seed + query_idx + (1_000_003 if cross_source else 0))
    selected = rng.choice(candidates, size=candidate_count, replace=False)
    return np.array(sorted(selected.tolist()), dtype=np.int64)


def topk_from_scores(scores: np.ndarray, candidate_indices: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(candidate_indices))
    if k <= 0:
        return np.array([], dtype=np.int64)
    selected = np.argpartition(-scores, kth=np.arange(k))[:k]
    selected = selected[np.argsort(-scores[selected])]
    return candidate_indices[selected]


def ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = ordinal_rank(x)
    ry = ordinal_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def evaluate_pair(
    candidate_features: np.ndarray,
    reference_features: np.ndarray,
    samples: List[Sample],
    query_indices: np.ndarray,
    top_ks: List[int],
    candidate_count: int,
    seed: int,
    cross_source: bool,
) -> Dict[str, float]:
    cand = l2_normalize(candidate_features.astype(np.float32))
    ref = l2_normalize(reference_features.astype(np.float32))
    max_k = max(top_ks)
    overlaps = {k: [] for k in top_ks}
    jaccards = {k: [] for k in top_ks}
    spearmans = []

    for query_idx in query_indices:
        pool = choose_candidate_indices(samples, int(query_idx), candidate_count, seed, cross_source)
        if len(pool) < 2:
            continue
        cand_scores = cand[int(query_idx)] @ cand[pool].T
        ref_scores = ref[int(query_idx)] @ ref[pool].T
        cand_top = topk_from_scores(cand_scores, pool, max_k)
        ref_top = topk_from_scores(ref_scores, pool, max_k)
        spearmans.append(spearman_corr(cand_scores, ref_scores))
        for k in top_ks:
            cand_set = set(cand_top[:k].tolist())
            ref_set = set(ref_top[:k].tolist())
            if not ref_set:
                continue
            inter = len(cand_set & ref_set)
            overlaps[k].append(inter / len(ref_set))
            union = len(cand_set | ref_set)
            jaccards[k].append(inter / union if union else 0.0)

    result: Dict[str, float] = {
        "spearman_mean": float(np.nanmean(spearmans)) if spearmans else float("nan"),
        "spearman_std": float(np.nanstd(spearmans)) if spearmans else float("nan"),
        "num_queries_evaluated": float(len(spearmans)),
    }
    for k in top_ks:
        result[f"overlap_at_{k}"] = float(np.mean(overlaps[k])) if overlaps[k] else float("nan")
        result[f"jaccard_at_{k}"] = float(np.mean(jaccards[k])) if jaccards[k] else float("nan")
    return result


def save_features(run_dir: Path, features: Dict[str, np.ndarray]) -> Dict[str, str]:
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, x in features.items():
        path = feature_dir / f"{name}.npy"
        np.save(path, x)
        paths[name] = str(path)
    return paths


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_root, run_dir = build_run_dir(args)
    print(f"[output] root={output_root} run_dir={run_dir}", flush=True)

    samples = load_samples(
        Path(args.input_jsonl),
        max_samples=args.max_samples,
        strategy=args.sample_strategy,
        seed=args.seed,
    )
    if len(samples) < 2:
        raise SystemExit("Need at least two samples.")
    write_samples(run_dir, samples)
    query_indices = choose_query_indices(samples, args.query_count, args.seed)
    top_ks = parse_int_list(args.top_k)

    hidden_x, hidden_specs = hidden_features(args, samples)
    ref_x, ref_specs = reference_features(args, samples)
    base_x, base_specs = baseline_features(args, samples)

    all_features = {**hidden_x, **ref_x, **base_x}
    all_specs = {**hidden_specs, **ref_specs, **base_specs}
    feature_paths = save_features(run_dir, all_features) if args.save_features else {}

    rows: List[Dict[str, object]] = []
    candidate_names = list(hidden_x.keys()) + list(base_x.keys())
    for reference_name, reference_matrix in ref_x.items():
        for candidate_name in candidate_names:
            print(f"[eval] candidate={candidate_name} reference={reference_name}", flush=True)
            for cross_source in [False, True]:
                metrics = evaluate_pair(
                    candidate_features=all_features[candidate_name],
                    reference_features=reference_matrix,
                    samples=samples,
                    query_indices=query_indices,
                    top_ks=top_ks,
                    candidate_count=args.candidate_count,
                    seed=args.seed,
                    cross_source=cross_source,
                )
                row = {
                    "candidate": candidate_name,
                    "candidate_family": all_specs[candidate_name].family,
                    "candidate_detail": all_specs[candidate_name].detail,
                    "candidate_normalize": all_specs[candidate_name].normalize,
                    "candidate_remove_top_pcs": all_specs[candidate_name].remove_top_pcs,
                    "reference": reference_name,
                    "reference_detail": all_specs[reference_name].detail,
                    "mode": "cross_source" if cross_source else "all",
                    "num_samples": len(samples),
                    "query_count": len(query_indices),
                    "candidate_count": args.candidate_count,
                    "feature_path": feature_paths.get(candidate_name, ""),
                    **metrics,
                }
                rows.append(row)

    fieldnames = [
        "candidate",
        "candidate_family",
        "candidate_detail",
        "candidate_normalize",
        "candidate_remove_top_pcs",
        "reference",
        "reference_detail",
        "mode",
        "num_samples",
        "query_count",
        "candidate_count",
        "num_queries_evaluated",
        "spearman_mean",
        "spearman_std",
    ]
    for k in top_ks:
        fieldnames.extend([f"overlap_at_{k}", f"jaccard_at_{k}"])
    fieldnames.append("feature_path")

    metrics_path = run_dir / "semantic_reference_metrics.csv"
    write_csv(metrics_path, rows, fieldnames)
    manifest = {
        "created_at_unix": int(time.time()),
        "args": vars(args),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "source_counts": source_counts(samples),
        "references": [asdict(ref_specs[name]) for name in ref_x],
        "candidates": [asdict(all_specs[name]) for name in candidate_names],
        "outputs": {
            "samples": str(run_dir / "samples.jsonl"),
            "metrics": str(metrics_path),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
