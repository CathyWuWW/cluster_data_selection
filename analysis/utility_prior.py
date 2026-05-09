"""
Build a non-invasive utility prior on the PMP grad_gamma vector.

Given an Exp2 utility CSV (rows aligned with cluster_id 0..K-1), produce a
length-K vector that is used to *initialize* grad_gamma at training start.
This intentionally does NOT modify PMP updates, the sampler, the loss, or
the gradient logic — it only shifts the starting point of the softmax.

Sampler relation (see data/cluster_dataset.ClusterWeightedSampler):

    w_k = softmax( -grad_gamma_k / T )

So if we want HIGH-utility clusters to get HIGHER initial sampling weight
("sign=positive"), we must initialize:

    grad_gamma_k = - alpha * normalized_utility_k

If we want low-utility clusters to be boosted instead, set sign="negative".

Supported aggregators (multi-ability -> scalar per cluster):
    sum | mean | max | weighted_sum
Supported normalizations (per-cluster scalar -> bounded scale):
    none | zscore | minmax | rank
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np


def _read_utility_csv(csv_path: str) -> tuple[List[str], np.ndarray]:
    """Load a utility CSV. Expect a 'cluster_id' column and >=1 ability columns."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"utility_prior: CSV not found: {csv_path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"utility_prior: empty CSV: {csv_path}")
    if "cluster_id" not in rows[0]:
        raise ValueError(
            f"utility_prior: CSV must have a 'cluster_id' column: {csv_path}"
        )
    ability_names = [c for c in rows[0].keys() if c != "cluster_id"]
    if not ability_names:
        raise ValueError(
            f"utility_prior: CSV has no ability columns besides cluster_id: {csv_path}"
        )

    ids = np.asarray([int(r["cluster_id"]) for r in rows], dtype=np.int64)
    expected = np.arange(ids.shape[0], dtype=np.int64)
    if not np.array_equal(ids, expected):
        raise ValueError(
            "utility_prior: CSV cluster_id column must be a contiguous "
            "range 0..K-1 with no gaps and no permutation."
        )
    matrix = np.asarray(
        [[float(r[a]) for a in ability_names] for r in rows], dtype=np.float64
    )
    return ability_names, matrix


def _select_columns(
    matrix: np.ndarray,
    ability_names: Sequence[str],
    selected: Sequence[str],
) -> tuple[np.ndarray, List[str]]:
    if not selected:
        return matrix, list(ability_names)
    indices = []
    for a in selected:
        if a not in ability_names:
            raise ValueError(
                f"utility_prior: ability '{a}' not in CSV columns {list(ability_names)}"
            )
        indices.append(ability_names.index(a))
    return matrix[:, indices], list(selected)


def _aggregate(
    matrix: np.ndarray,
    aggregator: str,
    weights: Sequence[float] | None,
) -> np.ndarray:
    """Reduce shape [K, A] -> [K] using one of the supported aggregators."""
    aggregator = aggregator.lower()
    K, A = matrix.shape
    if aggregator == "sum":
        return matrix.sum(axis=1)
    if aggregator == "mean":
        return matrix.mean(axis=1)
    if aggregator == "max":
        return matrix.max(axis=1)
    if aggregator == "weighted_sum":
        if not weights:
            return matrix.sum(axis=1)
        if len(weights) != A:
            raise ValueError(
                f"utility_prior: weights length {len(weights)} != n_abilities {A}"
            )
        w = np.asarray(weights, dtype=np.float64)
        if w.sum() <= 0:
            raise ValueError("utility_prior: weights must sum to a positive value")
        w = w / w.sum()
        return matrix @ w
    raise ValueError(
        f"utility_prior: unknown aggregator '{aggregator}'. "
        "Use one of: sum | mean | max | weighted_sum"
    )


def _normalize(vector: np.ndarray, normalize: str) -> np.ndarray:
    normalize = normalize.lower()
    if normalize == "none":
        return vector.astype(np.float64)
    if normalize == "zscore":
        mu = float(vector.mean())
        sd = float(vector.std())
        if sd < 1e-8:
            return np.zeros_like(vector, dtype=np.float64)
        return (vector - mu) / sd
    if normalize == "minmax":
        lo = float(vector.min())
        hi = float(vector.max())
        if hi - lo < 1e-8:
            return np.zeros_like(vector, dtype=np.float64)
        # Map to [-1, 1] so that the prior is centered.
        return 2.0 * (vector - lo) / (hi - lo) - 1.0
    if normalize == "rank":
        # Map ranks to roughly [-1, 1] keeping order.
        order = vector.argsort().argsort().astype(np.float64)
        K = vector.shape[0]
        if K <= 1:
            return np.zeros_like(vector, dtype=np.float64)
        return 2.0 * order / (K - 1) - 1.0
    raise ValueError(
        f"utility_prior: unknown normalize '{normalize}'. "
        "Use one of: none | zscore | minmax | rank"
    )


def load_utility_prior(
    csv_path: str,
    n_clusters: int,
    abilities: Sequence[str],
    weights: Sequence[float],
    aggregator: str,
    normalize: str,
    alpha: float,
    sign: str,
) -> np.ndarray:
    """
    Compute a length-`n_clusters` initial grad_gamma vector from a utility CSV.

    Args:
        csv_path:    path to utility CSV (rows aligned to cluster_id 0..K-1).
        n_clusters:  expected number of clusters in the run; must match CSV rows.
        abilities:   ability columns to use; empty = all non-id columns.
        weights:     per-ability weights (only used when aggregator='weighted_sum').
        aggregator:  sum | mean | max | weighted_sum
        normalize:   none | zscore | minmax | rank
        alpha:       prior strength scalar.
        sign:        "positive" => high utility -> high sampling weight.
                     "negative" => low utility  -> high sampling weight.

    Returns:
        np.ndarray of shape [n_clusters], dtype float32, the initial grad_gamma.
        If alpha == 0, returns zeros (i.e., behaves like default disabled prior).
    """
    if n_clusters <= 0:
        raise ValueError(f"utility_prior: n_clusters must be > 0, got {n_clusters}")

    ability_names, matrix = _read_utility_csv(csv_path)
    if matrix.shape[0] != n_clusters:
        raise ValueError(
            f"utility_prior: CSV has {matrix.shape[0]} clusters but the run "
            f"reports {n_clusters}. They must match for a 87-way prior."
        )

    selected_matrix, _ = _select_columns(matrix, ability_names, abilities)
    fused = _aggregate(selected_matrix, aggregator, weights)  # [K]
    normed = _normalize(fused, normalize)                     # [K]

    if alpha == 0.0:
        return np.zeros(n_clusters, dtype=np.float32)

    sign = sign.lower()
    if sign == "positive":
        # Boost high-utility clusters: smaller grad_gamma -> larger softmax weight.
        init = -alpha * normed
    elif sign == "negative":
        init = alpha * normed
    else:
        raise ValueError(
            f"utility_prior: unknown sign '{sign}'. Use 'positive' or 'negative'."
        )

    init = init - float(init.mean())  # zero-mean, harmless to softmax but cleaner
    return init.astype(np.float32)
