"""
Layer-2 representation comparison for cluster validation.

This script compares hidden-state representations with sentence embedding,
random, lexical, length, and source baselines on the same samples. It writes
CSV summaries that can be plotted directly and keeps the experiment independent
from the training loop.
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from nearest_neighbor_sanity import (
    RepresentationSpec,
    Sample,
    extract_hidden_features,
    get_num_layers,
    l2_normalize,
    length_correlation,
    load_model_and_tokenizer,
    load_samples,
    pairwise_topk_cosine,
    pca_summary,
    pool_hidden,
    random_same_source_baseline,
    remove_top_pcs,
    resolve_layer,
    source_counts,
    source_neighbor_rate,
    torch_dtype,
    write_samples,
)


@dataclass
class FeatureSpec:
    name: str
    family: str
    detail: str
    normalize: str = "none"
    remove_top_pcs: int = 0
    whiten: bool = False
    feature_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hidden, sentence, random, lexical, and source baselines."
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
    parser.add_argument("--top-k", type=int, default=10)
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
        "--hidden-layers",
        default="first,1,2,3,middle,-4,-3,-2,last",
        help="Comma-separated individual layer labels.",
    )
    parser.add_argument(
        "--hidden-layer-groups",
        default="early=first,1,2,3;middle=middle;late=-4,-3,-2;final=last",
        help="Semicolon-separated group_name=layer,layer specs. Empty disables groups.",
    )
    parser.add_argument(
        "--poolings",
        default="mean,last",
        help="Hidden-state poolings to compare: mean,last.",
    )
    parser.add_argument(
        "--normalize-options",
        default="l2",
        help="Comma-separated normalization variants for hidden features: raw,l2.",
    )
    parser.add_argument(
        "--remove-top-pcs",
        default="0,1,3",
        help="Comma-separated top-PC removal counts for hidden features.",
    )
    parser.add_argument(
        "--include-whitening",
        action="store_true",
        help="Also add PCA-whitened hidden mean-pooling variants.",
    )

    parser.add_argument(
        "--sentence-models",
        default="",
        help=(
            "Comma-separated sentence embedding models, e.g. "
            "intfloat/e5-small-v2,BAAI/bge-small-en-v1.5."
        ),
    )
    parser.add_argument(
        "--sentence-prefix",
        default="auto",
        help="Prefix for sentence models. Use auto for E5 passage prefix, or empty string.",
    )

    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--random-dim", type=int, default=768)
    parser.add_argument("--include-lexical", action="store_true")
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--include-length-source", action="store_true")

    parser.add_argument(
        "--kmeans-k",
        default="10,20,50",
        help="Comma-separated KMeans K values.",
    )
    parser.add_argument("--kmeans-n-init", type=int, default=5)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--kmeans-batch-size", type=int, default=1024)
    parser.add_argument("--silhouette-sample-size", type=int, default=1000)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
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
                f"kmeans-{slug(args.kmeans_k)}",
                f"seed{args.seed}",
            ]
        )
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return root, run_dir


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_layer_groups(raw: str, num_layers: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    if not raw.strip():
        return groups
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, layers = item.split("=", 1)
        groups[slug(name)] = [resolve_layer(x, num_layers) for x in parse_csv_list(layers)]
    return groups


def make_hidden_specs(args: argparse.Namespace, num_layers: int) -> Tuple[List[RepresentationSpec], Dict[str, List[str]]]:
    layers = [resolve_layer(x, num_layers) for x in parse_csv_list(args.hidden_layers)]
    poolings = [x.lower() for x in parse_csv_list(args.poolings)]
    groups = parse_layer_groups(args.hidden_layer_groups, num_layers)

    raw_specs: Dict[str, RepresentationSpec] = {}
    grouped_raw_names: Dict[str, List[str]] = {}

    for layer in layers:
        for pooling in poolings:
            name = f"hidden_layer{layer}_{pooling}"
            raw_specs[name] = RepresentationSpec(
                name=name,
                layer=layer,
                pooling=pooling,
                normalize=False,
                remove_top_pcs=0,
            )

    for group_name, group_layers in groups.items():
        for pooling in poolings:
            member_names = []
            for layer in group_layers:
                member_name = f"_groupbase_{group_name}_layer{layer}_{pooling}"
                raw_specs[member_name] = RepresentationSpec(
                    name=member_name,
                    layer=layer,
                    pooling=pooling,
                    normalize=False,
                    remove_top_pcs=0,
                )
                member_names.append(member_name)
            grouped_raw_names[f"hidden_group_{group_name}_{pooling}"] = member_names

    return list(raw_specs.values()), grouped_raw_names


def whiten_pca(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    scale = np.sqrt(max(centered.shape[0] - 1, 1)) / np.maximum(s, eps)
    return (centered @ vh.T * scale).astype(np.float32)


def transform_matrix(
    x: np.ndarray,
    normalize: str,
    pc_count: int,
    whiten: bool,
) -> np.ndarray:
    out = x.astype(np.float32)
    if pc_count > 0:
        out = remove_top_pcs(out, pc_count).astype(np.float32)
    if whiten:
        out = whiten_pca(out)
    if normalize == "l2":
        out = l2_normalize(out).astype(np.float32)
    return out


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
    raw_specs, grouped_raw_names = make_hidden_specs(args, num_layers)
    print(
        f"[hidden] model={args.hidden_model} num_layers={num_layers} raw_specs={len(raw_specs)}",
        flush=True,
    )
    raw = extract_hidden_features(model_args, samples, model, tokenizer, device, raw_specs)

    base_features: Dict[str, np.ndarray] = {}
    base_details: Dict[str, str] = {}
    for name, arr in raw.items():
        if name.startswith("_groupbase_"):
            continue
        base_features[name] = arr
        base_details[name] = name.replace("hidden_", "")
    for group_name, members in grouped_raw_names.items():
        base_features[group_name] = np.mean([raw[m] for m in members], axis=0).astype(np.float32)
        base_details[group_name] = group_name.replace("hidden_", "")

    normalize_options = [x.lower() for x in parse_csv_list(args.normalize_options)]
    pc_options = parse_int_list(args.remove_top_pcs)
    features: Dict[str, np.ndarray] = {}
    specs: Dict[str, FeatureSpec] = {}
    for base_name, arr in base_features.items():
        for norm in normalize_options:
            for pc_count in pc_options:
                suffix = norm
                if pc_count > 0:
                    suffix += f"_rm{pc_count}pc"
                name = f"{base_name}_{suffix}"
                features[name] = transform_matrix(arr, normalize=norm, pc_count=pc_count, whiten=False)
                specs[name] = FeatureSpec(
                    name=name,
                    family="hidden",
                    detail=base_details[base_name],
                    normalize=norm,
                    remove_top_pcs=pc_count,
                    whiten=False,
                )
            if args.include_whitening and "mean" in base_name:
                name = f"{base_name}_{norm}_whiten"
                features[name] = transform_matrix(arr, normalize=norm, pc_count=0, whiten=True)
                specs[name] = FeatureSpec(
                    name=name,
                    family="hidden",
                    detail=base_details[base_name],
                    normalize=norm,
                    remove_top_pcs=0,
                    whiten=True,
                )
    return features, specs


def sentence_prefix_for_model(model_name: str, prefix_arg: str) -> str:
    if prefix_arg == "auto":
        return "passage: " if "e5" in model_name.lower() else ""
    return prefix_arg


def extract_sentence_model_features(
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
                print(
                    f"[sentence] model={model_name} batch={batch_idx}/{total_batches}",
                    flush=True,
                )

    return l2_normalize(np.concatenate(features, axis=0).astype(np.float32))


def sentence_features(args: argparse.Namespace, samples: List[Sample]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    features: Dict[str, np.ndarray] = {}
    specs: Dict[str, FeatureSpec] = {}
    for model_name in parse_csv_list(args.sentence_models):
        name = f"sentence_{slug(model_name)}"
        print(f"[sentence] loading {model_name}", flush=True)
        features[name] = extract_sentence_model_features(args, model_name, samples)
        specs[name] = FeatureSpec(
            name=name,
            family="sentence",
            detail=model_name,
            normalize="l2",
        )
    return features, specs


def baseline_features(args: argparse.Namespace, samples: List[Sample]) -> Tuple[Dict[str, np.ndarray], Dict[str, FeatureSpec]]:
    features: Dict[str, np.ndarray] = {}
    specs: Dict[str, FeatureSpec] = {}
    rng = np.random.default_rng(args.seed)

    if args.include_random:
        name = f"random_normal_d{args.random_dim}"
        x = rng.normal(size=(len(samples), args.random_dim)).astype(np.float32)
        features[name] = l2_normalize(x).astype(np.float32)
        specs[name] = FeatureSpec(name=name, family="random", detail=f"dim={args.random_dim}", normalize="l2")

    if args.include_length_source:
        lengths = np.array([s.length_chars for s in samples], dtype=np.float32)
        length_z = (lengths - lengths.mean()) / max(float(lengths.std()), 1e-6)
        name = "length_scalar"
        features[name] = length_z[:, None].astype(np.float32)
        specs[name] = FeatureSpec(name=name, family="length", detail="zscored_length_chars")

        sources = sorted({s.source for s in samples})
        source_to_idx = {source: idx for idx, source in enumerate(sources)}
        x = np.zeros((len(samples), len(sources)), dtype=np.float32)
        for row, sample in enumerate(samples):
            x[row, source_to_idx[sample.source]] = 1.0
        name = "source_onehot"
        features[name] = x
        specs[name] = FeatureSpec(name=name, family="source", detail="source_label_onehot")

    if args.include_lexical:
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
        x = vectorizer.fit_transform([s.text for s in samples]).toarray().astype(np.float32)
        features[name] = l2_normalize(x).astype(np.float32)
        specs[name] = FeatureSpec(
            name=name,
            family="lexical",
            detail=f"word_tfidf_max_features={args.tfidf_max_features}",
            normalize="l2",
        )

    return features, specs


def select_query_indices(samples: List[Sample], query_count: int, seed: int) -> np.ndarray:
    count = min(query_count, len(samples))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(samples), size=count, replace=False))


def cluster_source_stats(labels: np.ndarray, samples: List[Sample]) -> Dict[str, float]:
    sources = sorted({s.source for s in samples})
    total = len(samples)
    weighted_purity = 0.0
    weighted_entropy = 0.0
    for cluster_id in sorted(set(labels.tolist())):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        counts = np.array([sum(samples[int(i)].source == src for i in idx) for src in sources], dtype=np.float64)
        probs = counts[counts > 0] / len(idx)
        purity = float(probs.max()) if len(probs) else 0.0
        entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
        weight = len(idx) / total
        weighted_purity += weight * purity
        weighted_entropy += weight * entropy
    return {
        "weighted_source_purity": weighted_purity,
        "weighted_source_entropy": weighted_entropy,
    }


def run_kmeans_metrics(
    features: np.ndarray,
    samples: List[Sample],
    k_values: Iterable[int],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    x = l2_normalize(features.astype(np.float32))
    rows: List[Dict[str, object]] = []
    for k in k_values:
        if k <= 1 or k >= len(samples):
            continue
        print(f"[kmeans] k={k}", flush=True)
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            batch_size=args.kmeans_batch_size,
        )
        labels = km.fit_predict(x)
        counts = np.bincount(labels, minlength=k)
        silhouette = float("nan")
        if len(set(labels.tolist())) > 1:
            sample_size = min(args.silhouette_sample_size, len(samples))
            silhouette = float(
                silhouette_score(
                    x,
                    labels,
                    metric="cosine",
                    sample_size=sample_size,
                    random_state=args.seed,
                )
            )
        source_stats = cluster_source_stats(labels, samples)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette_cosine": silhouette,
                "cluster_size_min": int(counts.min()),
                "cluster_size_max": int(counts.max()),
                "cluster_size_mean": float(counts.mean()),
                "cluster_size_std": float(counts.std()),
                **source_stats,
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_feature_files(run_dir: Path, features: Dict[str, np.ndarray], specs: Dict[str, FeatureSpec]) -> None:
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for name, x in features.items():
        path = feature_dir / f"{name}.npy"
        np.save(path, x)
        specs[name].feature_path = str(path)


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
    query_indices = select_query_indices(samples, args.query_count, args.seed)
    lengths = np.array([s.length_chars for s in samples], dtype=np.float32)
    random_source_rate = random_same_source_baseline(samples, query_indices)
    k_values = parse_int_list(args.kmeans_k)

    all_features: Dict[str, np.ndarray] = {}
    all_specs: Dict[str, FeatureSpec] = {}

    hidden_x, hidden_specs = hidden_features(args, samples)
    all_features.update(hidden_x)
    all_specs.update(hidden_specs)

    sent_x, sent_specs = sentence_features(args, samples)
    all_features.update(sent_x)
    all_specs.update(sent_specs)

    base_x, base_specs = baseline_features(args, samples)
    all_features.update(base_x)
    all_specs.update(base_specs)

    if args.save_features:
        save_feature_files(run_dir, all_features, all_specs)

    rep_rows: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []
    for name, x in all_features.items():
        spec = all_specs[name]
        print(f"[metrics] representation={name} shape={x.shape}", flush=True)
        neighbor_indices, _ = pairwise_topk_cosine(x, query_indices, args.top_k)
        same_rate = source_neighbor_rate(samples, query_indices, neighbor_indices)
        rep_row = {
            **asdict(spec),
            "num_samples": len(samples),
            "num_dims": int(x.shape[1]),
            "query_count": len(query_indices),
            "top_k": args.top_k,
            "same_source_neighbor_rate": same_rate,
            "same_source_lift_vs_random": same_rate / random_source_rate if random_source_rate > 0 else 0.0,
            "random_same_source_baseline": random_source_rate,
            **length_correlation(x, lengths),
            **pca_summary(x),
        }
        rep_rows.append(rep_row)

        for cluster_row in run_kmeans_metrics(x, samples, k_values, args):
            cluster_rows.append(
                {
                    "representation": name,
                    "family": spec.family,
                    "detail": spec.detail,
                    **cluster_row,
                }
            )

    rep_fields = [
        "name",
        "family",
        "detail",
        "normalize",
        "remove_top_pcs",
        "whiten",
        "feature_path",
        "num_samples",
        "num_dims",
        "query_count",
        "top_k",
        "same_source_neighbor_rate",
        "same_source_lift_vs_random",
        "random_same_source_baseline",
        "feature_norm_mean",
        "feature_norm_std",
        "length_chars_mean",
        "length_chars_std",
        "norm_length_corr",
        "pc1_variance_ratio",
        "pc2_variance_ratio",
        "pc3_variance_ratio",
        "pc4_variance_ratio",
        "pc5_variance_ratio",
    ]
    cluster_fields = [
        "representation",
        "family",
        "detail",
        "k",
        "inertia",
        "silhouette_cosine",
        "cluster_size_min",
        "cluster_size_max",
        "cluster_size_mean",
        "cluster_size_std",
        "weighted_source_purity",
        "weighted_source_entropy",
    ]
    representation_summary = run_dir / "representation_summary.csv"
    clustering_summary = run_dir / "clustering_summary.csv"
    write_csv(representation_summary, rep_rows, rep_fields)
    write_csv(clustering_summary, cluster_rows, cluster_fields)

    manifest = {
        "created_at_unix": int(time.time()),
        "args": vars(args),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "source_counts": source_counts(samples),
        "outputs": {
            "samples": str(run_dir / "samples.jsonl"),
            "representation_summary": str(representation_summary),
            "clustering_summary": str(clustering_summary),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
