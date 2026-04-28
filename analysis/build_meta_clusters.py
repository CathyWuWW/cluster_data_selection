"""
Build Experiment-3 meta-cluster assignments from Experiment-2 utility matrices.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build utility-driven or random meta-cluster assignments."
    )
    parser.add_argument(
        "--utility-csv",
        required=True,
        help="CSV matrix like utility_matrix_sum_delta.csv with cluster_id as first column.",
    )
    parser.add_argument(
        "--base-cluster-ids",
        required=True,
        help="Base cluster_ids_initial.npy aligned with the train dataset order.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for meta-cluster artifacts.",
    )
    parser.add_argument(
        "--n-meta-clusters",
        type=int,
        required=True,
        help="Number of meta-clusters to build.",
    )
    parser.add_argument(
        "--mode",
        choices=("utility", "random"),
        default="utility",
        help="Use KMeans on utility vectors or random regrouping of base clusters.",
    )
    parser.add_argument(
        "--normalize",
        choices=("zscore", "none"),
        default="zscore",
        help="Per-ability normalization before utility meta-clustering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_utility_matrix(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Utility matrix is empty: {path}")

    ability_names = [name for name in rows[0].keys() if name != "cluster_id"]
    if not ability_names:
        raise ValueError(f"No ability columns found in {path}")

    cluster_ids = np.asarray([int(row["cluster_id"]) for row in rows], dtype=np.int32)
    expected = np.arange(cluster_ids.shape[0], dtype=np.int32)
    if not np.array_equal(cluster_ids, expected):
        raise ValueError(
            "Utility matrix cluster_id column must be a contiguous range starting at 0."
        )

    matrix = np.asarray(
        [[float(row[ability]) for ability in ability_names] for row in rows],
        dtype=np.float32,
    )
    return cluster_ids, ability_names, matrix


def zscore_features(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (matrix - mean) / std
    return normalized.astype(np.float32), mean.reshape(-1), std.reshape(-1)


def build_random_regroup(n_base_clusters: int, n_meta_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_base_clusters)
    mapping = np.empty(n_base_clusters, dtype=np.int32)
    for i, base_cluster_id in enumerate(perm.tolist()):
        mapping[base_cluster_id] = i * n_meta_clusters // n_base_clusters
    return mapping


def build_utility_regroup(
    features: np.ndarray,
    n_meta_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(
        n_clusters=n_meta_clusters,
        random_state=seed,
        n_init=20,
        max_iter=300,
    )
    labels = kmeans.fit_predict(features)
    return labels.astype(np.int32), kmeans.cluster_centers_.astype(np.float32)


def invert_mapping(labels: np.ndarray) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        out.setdefault(int(label), []).append(int(idx))
    return out


def build_assignment_payload(
    meta_cluster_ids: np.ndarray,
    base_to_meta: np.ndarray,
    mode: str,
    normalize: str,
    n_meta_clusters: int,
    ability_names: List[str],
    raw_matrix: np.ndarray,
    normalized_matrix: np.ndarray,
    center_matrix: np.ndarray | None,
    seed: int,
) -> dict:
    cluster_to_samples = invert_mapping(meta_cluster_ids)
    cluster_sizes = {int(cid): len(samples) for cid, samples in cluster_to_samples.items()}
    sample_to_cluster = {
        int(sample_id): int(cluster_id)
        for sample_id, cluster_id in enumerate(meta_cluster_ids.tolist())
    }

    payload = {
        "step": 0,
        "tag": f"meta_{mode}",
        "mode": mode,
        "normalize": normalize,
        "seed": seed,
        "n_samples": int(meta_cluster_ids.shape[0]),
        "n_clusters": int(n_meta_clusters),
        "cluster_sizes": cluster_sizes,
        "sample_to_cluster": sample_to_cluster,
        "cluster_to_samples": cluster_to_samples,
        "base_cluster_to_meta_cluster": {
            int(base_cluster_id): int(meta_cluster_id)
            for base_cluster_id, meta_cluster_id in enumerate(base_to_meta.tolist())
        },
        "abilities": ability_names,
        "utility_matrix_raw": raw_matrix.tolist(),
        "utility_matrix_normalized": normalized_matrix.tolist(),
    }
    if center_matrix is not None:
        payload["meta_cluster_centers"] = center_matrix.tolist()
    return payload


def write_matrix_csv(path: Path, ability_names: List[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", *ability_names])
        for cluster_id, row in enumerate(matrix.tolist()):
            writer.writerow([cluster_id, *row])


def write_base_to_meta_csv(path: Path, base_to_meta: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base_cluster_id", "meta_cluster_id"])
        for base_cluster_id, meta_cluster_id in enumerate(base_to_meta.tolist()):
            writer.writerow([base_cluster_id, meta_cluster_id])


def write_meta_summary_csv(
    path: Path,
    ability_names: List[str],
    base_to_meta: np.ndarray,
    sample_meta_ids: np.ndarray,
    raw_matrix: np.ndarray,
    normalized_matrix: np.ndarray,
) -> None:
    cluster_to_base = invert_mapping(base_to_meta)
    sample_counts = np.bincount(sample_meta_ids, minlength=int(base_to_meta.max()) + 1)

    fieldnames = [
        "meta_cluster_id",
        "n_base_clusters",
        "n_samples",
        "base_cluster_ids",
        *[f"mean_raw_{ability}" for ability in ability_names],
        *[f"mean_z_{ability}" for ability in ability_names],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta_cluster_id, base_cluster_ids in sorted(cluster_to_base.items()):
            base_idx = np.asarray(base_cluster_ids, dtype=np.int32)
            raw_mean = raw_matrix[base_idx].mean(axis=0)
            z_mean = normalized_matrix[base_idx].mean(axis=0)
            row = {
                "meta_cluster_id": int(meta_cluster_id),
                "n_base_clusters": int(len(base_cluster_ids)),
                "n_samples": int(sample_counts[int(meta_cluster_id)]),
                "base_cluster_ids": " ".join(str(x) for x in base_cluster_ids),
            }
            row.update(
                {
                    f"mean_raw_{ability}": float(raw_mean[i])
                    for i, ability in enumerate(ability_names)
                }
            )
            row.update(
                {
                    f"mean_z_{ability}": float(z_mean[i])
                    for i, ability in enumerate(ability_names)
                }
            )
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    utility_csv = Path(args.utility_csv)
    base_cluster_ids_path = Path(args.base_cluster_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cluster_ids = np.asarray(np.load(base_cluster_ids_path), dtype=np.int32).reshape(-1)
    if base_cluster_ids.size == 0:
        raise ValueError(f"Base cluster ids are empty: {base_cluster_ids_path}")

    cluster_ids, ability_names, raw_matrix = load_utility_matrix(utility_csv)
    n_base_clusters = raw_matrix.shape[0]
    if int(base_cluster_ids.max()) >= n_base_clusters:
        raise ValueError(
            "Base cluster ids reference a cluster index outside the utility matrix."
        )
    if args.n_meta_clusters <= 1 or args.n_meta_clusters > n_base_clusters:
        raise ValueError(
            f"n-meta-clusters must be in [2, {n_base_clusters}], got {args.n_meta_clusters}."
        )

    if args.normalize == "zscore":
        normalized_matrix, mean, std = zscore_features(raw_matrix)
    else:
        normalized_matrix = raw_matrix.astype(np.float32)
        mean = raw_matrix.mean(axis=0)
        std = raw_matrix.std(axis=0)

    if args.mode == "utility":
        base_to_meta, center_matrix = build_utility_regroup(
            features=normalized_matrix,
            n_meta_clusters=args.n_meta_clusters,
            seed=args.seed,
        )
    else:
        base_to_meta = build_random_regroup(
            n_base_clusters=n_base_clusters,
            n_meta_clusters=args.n_meta_clusters,
            seed=args.seed,
        )
        center_matrix = None

    meta_cluster_ids = base_to_meta[base_cluster_ids]
    payload = build_assignment_payload(
        meta_cluster_ids=meta_cluster_ids,
        base_to_meta=base_to_meta,
        mode=args.mode,
        normalize=args.normalize,
        n_meta_clusters=args.n_meta_clusters,
        ability_names=ability_names,
        raw_matrix=raw_matrix,
        normalized_matrix=normalized_matrix,
        center_matrix=center_matrix,
        seed=args.seed,
    )

    np.save(out_dir / "meta_cluster_ids.npy", meta_cluster_ids.astype(np.int32))
    with (out_dir / "meta_cluster_assignments.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    write_base_to_meta_csv(out_dir / "base_cluster_to_meta_cluster.csv", base_to_meta)
    write_matrix_csv(out_dir / "utility_matrix_raw.csv", ability_names, raw_matrix)
    write_matrix_csv(out_dir / "utility_matrix_normalized.csv", ability_names, normalized_matrix)
    write_meta_summary_csv(
        out_dir / "meta_cluster_summary.csv",
        ability_names=ability_names,
        base_to_meta=base_to_meta,
        sample_meta_ids=meta_cluster_ids,
        raw_matrix=raw_matrix,
        normalized_matrix=normalized_matrix,
    )

    metadata = {
        "mode": args.mode,
        "normalize": args.normalize,
        "seed": args.seed,
        "n_base_clusters": int(n_base_clusters),
        "n_meta_clusters": int(args.n_meta_clusters),
        "abilities": ability_names,
        "utility_csv": str(utility_csv),
        "base_cluster_ids": str(base_cluster_ids_path),
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": std.astype(np.float32).tolist(),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {args.mode} meta-cluster artifacts to {out_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
