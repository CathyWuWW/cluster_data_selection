"""
Nearest-neighbor sanity check for representation quality.

This script reads a JSONL subset exported by export_slimpajama_subset.py,
extracts one or more representations, and writes audit-friendly nearest
neighbor tables for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Sample:
    sample_index: int
    original_index: int
    sample_id: str
    source: str
    text: str
    length_chars: int
    length_tokens: int = 0


@dataclass
class RepresentationSpec:
    name: str
    layer: int
    pooling: str
    normalize: bool
    remove_top_pcs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export nearest-neighbor sanity tables for hidden representations."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--out-dir",
        required=True,
        help=(
            "Experiment output root. By default, a parameterized run subdirectory "
            "is created inside this directory."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run subdirectory name. If omitted, one is generated from key args.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write directly into --out-dir, preserving the older output layout.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument(
        "--sample-strategy",
        choices=["first", "random", "stratified"],
        default="first",
        help=(
            "How to select max-samples from the JSONL file. 'first' preserves "
            "the old behavior; 'random' samples across the whole file; "
            "'stratified' samples approximately evenly by source."
        ),
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
    parser.add_argument(
        "--attn-impl",
        default=None,
        help="Optional transformers attn_implementation, e.g. sdpa or flash_attention_2.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--layers",
        default="middle,last",
        help="Comma-separated layers: first,middle,last or integer indices.",
    )
    parser.add_argument(
        "--poolings",
        default="mean,last",
        help="Comma-separated pooling methods: mean,last.",
    )
    parser.add_argument(
        "--normalize-options",
        default="raw,l2",
        help="Comma-separated normalization variants: raw,l2.",
    )
    parser.add_argument(
        "--remove-top-pcs",
        default="0",
        help="Comma-separated top-PC removal counts, e.g. 0,1,3.",
    )
    parser.add_argument(
        "--representations",
        default="",
        help=(
            "Optional explicit specs separated by ';'. Format: "
            "name:layer:pooling:normalize:remove_top_pcs, e.g. mid_mean_l2:middle:mean:l2:0."
        ),
    )
    parser.add_argument(
        "--text-preview-chars",
        type=int,
        default=500,
        help="Truncate query/neighbor text in audit outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print feature extraction progress every N batches. Set 0 to disable.",
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    value = value.strip().lower()
    chars = []
    for ch in value:
        if ch.isalnum():
            chars.append(ch)
        elif ch in {".", "-", "_"}:
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return _slug(args.run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = _slug(args.model.split("/")[-1])
    sample_size = "all" if args.max_samples <= 0 else str(args.max_samples)
    parts = [
        timestamp,
        model,
        f"n{sample_size}",
        args.sample_strategy,
        f"q{args.query_count}",
        f"k{args.top_k}",
        f"len{args.max_length}",
        f"layers-{_slug(args.layers)}",
        f"pool-{_slug(args.poolings)}",
        f"norm-{_slug(args.normalize_options)}",
        f"pc-{_slug(args.remove_top_pcs)}",
        f"seed{args.seed}",
    ]
    return "_".join(parts)


def resolve_output_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_root = Path(args.out_dir)
    if args.flat_output:
        run_dir = output_root
    else:
        run_dir = output_root / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    return output_root, run_dir


def _read_all_samples(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            sample_id = str(obj.get("sample_id", idx))
            source = str(obj.get("source", "unknown"))
            text = text.strip()
            samples.append(
                Sample(
                    sample_index=len(samples),
                    original_index=idx,
                    sample_id=sample_id,
                    source=source,
                    text=text,
                    length_chars=len(text),
                )
            )
    return samples


def _reindex_samples(samples: List[Sample]) -> List[Sample]:
    for idx, sample in enumerate(samples):
        sample.sample_index = idx
    return samples


def _select_random(samples: List[Sample], max_samples: int, seed: int) -> List[Sample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return _reindex_samples(samples)
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(samples), size=max_samples, replace=False))
    return _reindex_samples([samples[int(i)] for i in selected_indices])


def _select_stratified(samples: List[Sample], max_samples: int, seed: int) -> List[Sample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return _reindex_samples(samples)

    by_source: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        by_source.setdefault(sample.source, []).append(idx)

    sources = sorted(by_source)
    base = max_samples // len(sources)
    remainder = max_samples % len(sources)
    quotas = {source: base for source in sources}
    for source in sources[:remainder]:
        quotas[source] += 1

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    underfilled = 0
    for source in sources:
        indices = by_source[source]
        quota = quotas[source]
        if len(indices) <= quota:
            selected.extend(indices)
            underfilled += quota - len(indices)
        else:
            selected.extend(rng.choice(indices, size=quota, replace=False).tolist())

    if underfilled > 0 and len(selected) < max_samples:
        selected_set = set(selected)
        remaining = [idx for idx in range(len(samples)) if idx not in selected_set]
        take = min(underfilled, len(remaining), max_samples - len(selected))
        if take > 0:
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    selected = sorted(selected[:max_samples])
    return _reindex_samples([samples[idx] for idx in selected])


def load_samples(path: Path, max_samples: int, strategy: str, seed: int) -> List[Sample]:
    samples = _read_all_samples(path)
    if strategy == "first":
        if max_samples > 0:
            samples = samples[:max_samples]
        return _reindex_samples(samples)
    if strategy == "random":
        return _select_random(samples, max_samples, seed)
    if strategy == "stratified":
        return _select_stratified(samples, max_samples, seed)
    raise ValueError(f"Unknown sample strategy: {strategy}")


def resolve_layer(layer: str, num_layers: int) -> int:
    layer = layer.strip().lower()
    if layer == "first":
        return 0
    if layer == "middle":
        return num_layers // 2
    if layer == "last":
        return num_layers - 1
    value = int(layer)
    if value < 0:
        value = num_layers + value
    if value < 0 or value >= num_layers:
        raise ValueError(f"Layer {layer} resolved to {value}, outside [0, {num_layers - 1}]")
    return value


def build_specs(args: argparse.Namespace, num_layers: int) -> List[RepresentationSpec]:
    if args.representations:
        specs: List[RepresentationSpec] = []
        for raw_spec in args.representations.split(";"):
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

    layers = [resolve_layer(x, num_layers) for x in args.layers.split(",") if x.strip()]
    poolings = [x.strip().lower() for x in args.poolings.split(",") if x.strip()]
    norm_options = [x.strip().lower() for x in args.normalize_options.split(",") if x.strip()]
    pc_options = [int(x.strip()) for x in args.remove_top_pcs.split(",") if x.strip()]

    specs = []
    for layer in layers:
        layer_label = f"layer{layer}"
        for pooling in poolings:
            for norm in norm_options:
                for remove_top_pcs in pc_options:
                    normalize = norm == "l2"
                    suffix = "l2" if normalize else "raw"
                    if remove_top_pcs > 0:
                        suffix += f"_rm{remove_top_pcs}pc"
                    specs.append(
                        RepresentationSpec(
                            name=f"{layer_label}_{pooling}_{suffix}",
                            layer=layer,
                            pooling=pooling,
                            normalize=normalize,
                            remove_top_pcs=remove_top_pcs,
                        )
                    )
    return specs


def torch_dtype(dtype: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_num_layers(model) -> int:
    config = getattr(model, "config", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine model.config.num_hidden_layers")
    return int(num_layers)


def pool_hidden(hidden, attention_mask, pooling: str):
    import torch

    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1)
        return (hidden * mask).sum(dim=1) / denom
    if pooling == "last":
        lengths = attention_mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, lengths - 1]
    raise ValueError(f"Unknown pooling: {pooling}")


def extract_hidden_features(
    args: argparse.Namespace,
    samples: List[Sample],
    model,
    tokenizer,
    device,
    specs: List[RepresentationSpec],
) -> Dict[str, np.ndarray]:
    import torch

    by_layer: Dict[int, List[RepresentationSpec]] = {}
    for spec in specs:
        by_layer.setdefault(spec.layer, []).append(spec)

    features: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in specs}
    total_batches = math.ceil(len(samples) / args.batch_size)
    started_at = time.time()

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(samples), args.batch_size), start=1):
            batch_samples = samples[start : start + args.batch_size]
            texts = [s.text for s in batch_samples]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            for offset, sample in enumerate(batch_samples):
                sample.length_tokens = int(encoded["attention_mask"][offset].sum().item())

            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[1:]
            for layer_idx, layer_specs in by_layer.items():
                hidden = hidden_states[layer_idx]
                for spec in layer_specs:
                    pooled = pool_hidden(hidden, encoded["attention_mask"], spec.pooling)
                    features[spec.name].append(pooled.cpu().float().numpy())

            if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                print(
                    f"[features] batch={batch_idx}/{total_batches} "
                    f"samples={min(start + args.batch_size, len(samples))}/{len(samples)} "
                    f"rate={min(start + args.batch_size, len(samples))/elapsed:.1f}/s",
                    flush=True,
                )

    return {name: np.concatenate(parts, axis=0) for name, parts in features.items()}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def remove_top_pcs(x: np.ndarray, n_components: int) -> np.ndarray:
    if n_components <= 0:
        return x
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pcs = vh[:n_components]
    return centered - centered @ pcs.T @ pcs


def transform_features(raw_features: Dict[str, np.ndarray], specs: List[RepresentationSpec]) -> Dict[str, np.ndarray]:
    transformed = {}
    for spec in specs:
        x = raw_features[spec.name].astype(np.float32)
        if spec.remove_top_pcs > 0:
            x = remove_top_pcs(x, spec.remove_top_pcs).astype(np.float32)
        if spec.normalize:
            x = l2_normalize(x).astype(np.float32)
        transformed[spec.name] = x
    return transformed


def pairwise_topk_cosine(features: np.ndarray, query_indices: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = l2_normalize(features.astype(np.float32))
    q = x[query_indices]
    sim = q @ x.T
    for row, query_idx in enumerate(query_indices):
        sim[row, query_idx] = -np.inf
    k = min(top_k, features.shape[0] - 1)
    idx = np.argpartition(-sim, kth=np.arange(k), axis=1)[:, :k]
    row_order = np.arange(idx.shape[0])[:, None]
    sorted_order = np.argsort(-sim[row_order, idx], axis=1)
    idx = idx[row_order, sorted_order]
    distances = 1.0 - sim[row_order, idx]
    return idx, distances


def length_correlation(features: np.ndarray, lengths: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(features, axis=1)
    if np.std(norms) == 0 or np.std(lengths) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(norms, lengths)[0, 1])
    return {
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
        "length_chars_mean": float(lengths.mean()),
        "length_chars_std": float(lengths.std()),
        "norm_length_corr": corr,
    }


def pca_summary(features: np.ndarray, max_components: int = 5) -> Dict[str, float]:
    x = features.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    result = {}
    for i in range(min(max_components, len(var))):
        result[f"pc{i+1}_variance_ratio"] = float(var[i] / total) if total > 0 else 0.0
    return result


def source_neighbor_rate(samples: List[Sample], query_indices: np.ndarray, neighbor_indices: np.ndarray) -> float:
    matches = 0
    total = 0
    for row, query_idx in enumerate(query_indices):
        q_source = samples[int(query_idx)].source
        for neighbor_idx in neighbor_indices[row]:
            matches += int(samples[int(neighbor_idx)].source == q_source)
            total += 1
    return matches / total if total else 0.0


def source_counts(samples: List[Sample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.source] = counts.get(sample.source, 0) + 1
    return dict(sorted(counts.items()))


def random_same_source_baseline(samples: List[Sample], query_indices: np.ndarray) -> float:
    if len(samples) <= 1 or len(query_indices) == 0:
        return 0.0
    counts = source_counts(samples)
    rates = []
    for query_idx in query_indices:
        query = samples[int(query_idx)]
        same_source_candidates = max(counts.get(query.source, 0) - 1, 0)
        rates.append(same_source_candidates / (len(samples) - 1))
    return float(np.mean(rates)) if rates else 0.0


def preview(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def write_samples(out_dir: Path, samples: List[Sample]) -> None:
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def write_neighbors(
    out_dir: Path,
    spec: RepresentationSpec,
    samples: List[Sample],
    query_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    distances: np.ndarray,
    text_preview_chars: int,
) -> None:
    rep_dir = out_dir / "nearest_neighbors" / spec.name
    rep_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = rep_dir / "neighbors.jsonl"
    csv_path = rep_dir / "neighbors.csv"
    fieldnames = [
        "representation",
        "query_rank",
        "query_index",
        "query_original_index",
        "query_sample_id",
        "query_source",
        "query_length_chars",
        "query_length_tokens",
        "query_text",
        "neighbor_rank",
        "neighbor_index",
        "neighbor_original_index",
        "neighbor_sample_id",
        "neighbor_source",
        "neighbor_length_chars",
        "neighbor_length_tokens",
        "neighbor_text",
        "distance",
        "same_source",
    ]
    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for query_rank, query_idx in enumerate(query_indices):
            query = samples[int(query_idx)]
            for neighbor_rank, neighbor_idx in enumerate(neighbor_indices[query_rank], start=1):
                neighbor = samples[int(neighbor_idx)]
                row = {
                    "representation": spec.name,
                    "query_rank": query_rank,
                    "query_index": query.sample_index,
                    "query_original_index": query.original_index,
                    "query_sample_id": query.sample_id,
                    "query_source": query.source,
                    "query_length_chars": query.length_chars,
                    "query_length_tokens": query.length_tokens,
                    "query_text": preview(query.text, text_preview_chars),
                    "neighbor_rank": neighbor_rank,
                    "neighbor_index": neighbor.sample_index,
                    "neighbor_original_index": neighbor.original_index,
                    "neighbor_sample_id": neighbor.sample_id,
                    "neighbor_source": neighbor.source,
                    "neighbor_length_chars": neighbor.length_chars,
                    "neighbor_length_tokens": neighbor.length_tokens,
                    "neighbor_text": preview(neighbor.text, text_preview_chars),
                    "distance": float(distances[query_rank, neighbor_rank - 1]),
                    "same_source": query.source == neighbor.source,
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.writerow(row)


def write_feature_files(out_dir: Path, features: Dict[str, np.ndarray]) -> Dict[str, str]:
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in features.items():
        path = feature_dir / f"{name}.npy"
        np.save(path, array)
        paths[name] = str(path)
    return paths


def write_summary_csv(
    out_dir: Path,
    metrics: Dict[str, object],
    feature_paths: Dict[str, str],
) -> str:
    summary_path = out_dir / "summary.csv"
    fieldnames = [
        "representation",
        "layer",
        "pooling",
        "normalize",
        "remove_top_pcs",
        "same_source_neighbor_rate",
        "same_source_lift_vs_random",
        "random_same_source_baseline",
        "norm_length_corr",
        "feature_norm_mean",
        "feature_norm_std",
        "pc1_variance_ratio",
        "pc2_variance_ratio",
        "pc3_variance_ratio",
        "pc4_variance_ratio",
        "pc5_variance_ratio",
        "neighbors_csv",
        "feature_path",
    ]
    baseline = float(metrics.get("random_same_source_baseline", 0.0))
    reps = metrics.get("representations", {})
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, rep_metrics in reps.items():
            same_source_rate = float(rep_metrics.get("same_source_neighbor_rate", 0.0))
            lift = same_source_rate / baseline if baseline > 0 else 0.0
            row = {
                "representation": name,
                "layer": rep_metrics.get("layer"),
                "pooling": rep_metrics.get("pooling"),
                "normalize": rep_metrics.get("normalize"),
                "remove_top_pcs": rep_metrics.get("remove_top_pcs"),
                "same_source_neighbor_rate": same_source_rate,
                "same_source_lift_vs_random": lift,
                "random_same_source_baseline": baseline,
                "norm_length_corr": rep_metrics.get("norm_length_corr"),
                "feature_norm_mean": rep_metrics.get("feature_norm_mean"),
                "feature_norm_std": rep_metrics.get("feature_norm_std"),
                "pc1_variance_ratio": rep_metrics.get("pc1_variance_ratio"),
                "pc2_variance_ratio": rep_metrics.get("pc2_variance_ratio"),
                "pc3_variance_ratio": rep_metrics.get("pc3_variance_ratio"),
                "pc4_variance_ratio": rep_metrics.get("pc4_variance_ratio"),
                "pc5_variance_ratio": rep_metrics.get("pc5_variance_ratio"),
                "neighbors_csv": str(out_dir / "nearest_neighbors" / name / "neighbors.csv"),
                "feature_path": feature_paths.get(name, ""),
            }
            writer.writerow(row)
    return str(summary_path)


def _summary_rows(
    run_dir: Path,
    metrics: Dict[str, object],
    feature_paths: Dict[str, str],
) -> List[Dict[str, object]]:
    baseline = float(metrics.get("random_same_source_baseline", 0.0))
    reps = metrics.get("representations", {})
    rows: List[Dict[str, object]] = []
    for name, rep_metrics in reps.items():
        same_source_rate = float(rep_metrics.get("same_source_neighbor_rate", 0.0))
        rows.append(
            {
                "run_dir": str(run_dir),
                "model": metrics.get("model"),
                "input_jsonl": metrics.get("input_jsonl"),
                "num_samples": metrics.get("num_samples"),
                "sample_strategy": metrics.get("sample_strategy"),
                "query_count": metrics.get("query_count"),
                "top_k": metrics.get("top_k"),
                "seed": metrics.get("seed"),
                "source_counts_json": json.dumps(
                    metrics.get("source_counts", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "representation": name,
                "layer": rep_metrics.get("layer"),
                "pooling": rep_metrics.get("pooling"),
                "normalize": rep_metrics.get("normalize"),
                "remove_top_pcs": rep_metrics.get("remove_top_pcs"),
                "same_source_neighbor_rate": same_source_rate,
                "same_source_lift_vs_random": same_source_rate / baseline if baseline > 0 else 0.0,
                "random_same_source_baseline": baseline,
                "norm_length_corr": rep_metrics.get("norm_length_corr"),
                "feature_norm_mean": rep_metrics.get("feature_norm_mean"),
                "feature_norm_std": rep_metrics.get("feature_norm_std"),
                "pc1_variance_ratio": rep_metrics.get("pc1_variance_ratio"),
                "pc2_variance_ratio": rep_metrics.get("pc2_variance_ratio"),
                "pc3_variance_ratio": rep_metrics.get("pc3_variance_ratio"),
                "pc4_variance_ratio": rep_metrics.get("pc4_variance_ratio"),
                "pc5_variance_ratio": rep_metrics.get("pc5_variance_ratio"),
                "summary_csv": str(run_dir / "summary.csv"),
                "neighbors_csv": str(run_dir / "nearest_neighbors" / name / "neighbors.csv"),
                "feature_path": feature_paths.get(name, ""),
            }
        )
    return rows


def update_runs_summary_csv(
    output_root: Path,
    run_dir: Path,
    metrics: Dict[str, object],
    feature_paths: Dict[str, str],
) -> str:
    summary_path = output_root / "runs_summary.csv"
    fieldnames = [
        "run_dir",
        "model",
        "input_jsonl",
        "num_samples",
        "sample_strategy",
        "query_count",
        "top_k",
        "seed",
        "source_counts_json",
        "representation",
        "layer",
        "pooling",
        "normalize",
        "remove_top_pcs",
        "same_source_neighbor_rate",
        "same_source_lift_vs_random",
        "random_same_source_baseline",
        "norm_length_corr",
        "feature_norm_mean",
        "feature_norm_std",
        "pc1_variance_ratio",
        "pc2_variance_ratio",
        "pc3_variance_ratio",
        "pc4_variance_ratio",
        "pc5_variance_ratio",
        "summary_csv",
        "neighbors_csv",
        "feature_path",
    ]

    rows: List[Dict[str, object]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(row for row in reader if row.get("run_dir") != str(run_dir))
    rows.extend(_summary_rows(run_dir, metrics, feature_paths))

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(summary_path)


def main() -> None:
    args = parse_args()
    output_root, out_dir = resolve_output_dirs(args)
    print(f"[output] root={output_root} run_dir={out_dir}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    samples = load_samples(
        Path(args.input_jsonl),
        max_samples=args.max_samples,
        strategy=args.sample_strategy,
        seed=args.seed,
    )
    if len(samples) < 2:
        raise SystemExit("Need at least two samples for nearest-neighbor analysis.")

    model, tokenizer, device = load_model_and_tokenizer(args)
    num_layers = get_num_layers(model)
    specs = build_specs(args, num_layers)
    print(
        f"[start] samples={len(samples)} model={args.model} "
        f"num_layers={num_layers} specs={[s.name for s in specs]}",
        flush=True,
    )

    query_count = min(args.query_count, len(samples))
    rng = np.random.default_rng(args.seed)
    query_indices = np.sort(rng.choice(len(samples), size=query_count, replace=False))

    raw_features = extract_hidden_features(args, samples, model, tokenizer, device, specs)
    features = transform_features(raw_features, specs)
    feature_paths = write_feature_files(out_dir, features)
    write_samples(out_dir, samples)

    lengths = np.array([s.length_chars for s in samples], dtype=np.float32)
    metrics = {
        "input_jsonl": args.input_jsonl,
        "model": args.model,
        "output_root": str(output_root),
        "run_dir": str(out_dir),
        "num_samples": len(samples),
        "sample_strategy": args.sample_strategy,
        "source_counts": source_counts(samples),
        "query_count": query_count,
        "top_k": args.top_k,
        "seed": args.seed,
        "random_same_source_baseline": random_same_source_baseline(samples, query_indices),
        "representations": {},
    }

    for spec in specs:
        print(f"[neighbors] representation={spec.name}", flush=True)
        x = features[spec.name]
        neighbor_indices, distances = pairwise_topk_cosine(x, query_indices, args.top_k)
        write_neighbors(
            out_dir=out_dir,
            spec=spec,
            samples=samples,
            query_indices=query_indices,
            neighbor_indices=neighbor_indices,
            distances=distances,
            text_preview_chars=args.text_preview_chars,
        )
        rep_metrics = {
            **asdict(spec),
            **length_correlation(x, lengths),
            **pca_summary(x),
            "same_source_neighbor_rate": source_neighbor_rate(samples, query_indices, neighbor_indices),
            "feature_path": feature_paths[spec.name],
        }
        metrics["representations"][spec.name] = rep_metrics

    summary_path = write_summary_csv(out_dir, metrics, feature_paths)
    runs_summary_path = update_runs_summary_csv(output_root, out_dir, metrics, feature_paths)
    manifest = {
        "created_at_unix": int(time.time()),
        "args": vars(args),
        "output_root": str(output_root),
        "run_dir": str(out_dir),
        "num_layers": num_layers,
        "specs": [asdict(spec) for spec in specs],
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "metrics": str(out_dir / "metrics.json"),
            "summary": summary_path,
            "runs_summary": runs_summary_path,
            "features": feature_paths,
            "nearest_neighbors_dir": str(out_dir / "nearest_neighbors"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
