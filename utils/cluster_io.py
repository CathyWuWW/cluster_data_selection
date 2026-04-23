"""
Utilities for loading precomputed cluster assignments.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable

import numpy as np


def _cluster_ids_from_sample_to_cluster(sample_to_cluster: Dict) -> np.ndarray:
    if not sample_to_cluster:
        raise ValueError("sample_to_cluster is empty.")

    normalized = {}
    for raw_sample_id, raw_cluster_id in sample_to_cluster.items():
        normalized[int(raw_sample_id)] = int(raw_cluster_id)

    sample_ids = sorted(normalized.keys())
    if sample_ids != list(range(len(sample_ids))):
        raise ValueError(
            "sample_to_cluster keys must form a contiguous range starting at 0."
        )

    cluster_ids = np.empty(len(sample_ids), dtype=np.int32)
    for sample_id, cluster_id in normalized.items():
        cluster_ids[sample_id] = cluster_id
    return cluster_ids


def _cluster_ids_from_cluster_to_samples(cluster_to_samples: Dict) -> np.ndarray:
    if not cluster_to_samples:
        raise ValueError("cluster_to_samples is empty.")

    sample_to_cluster = {}
    for raw_cluster_id, sample_ids in cluster_to_samples.items():
        cluster_id = int(raw_cluster_id)
        if not isinstance(sample_ids, Iterable):
            raise ValueError("cluster_to_samples values must be iterables of sample ids.")
        for sample_id in sample_ids:
            sample_idx = int(sample_id)
            if sample_idx in sample_to_cluster:
                raise ValueError(
                    f"Duplicate sample id {sample_idx} found in cluster_to_samples."
                )
            sample_to_cluster[sample_idx] = cluster_id
    return _cluster_ids_from_sample_to_cluster(sample_to_cluster)


def load_precomputed_cluster_ids(path: str) -> np.ndarray:
    """
    Load precomputed cluster ids from disk.

    Supported formats:
      - `cluster_ids*.npy`: numpy int array aligned with dataset order
      - `cluster_assignments*.json`: payload containing `sample_to_cluster`
        or `cluster_to_samples`
      - plain JSON list of cluster ids
    """
    if not path:
        raise ValueError("Precomputed cluster path is empty.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Precomputed cluster file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        cluster_ids = np.load(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            cluster_ids = np.asarray(payload, dtype=np.int32)
        elif isinstance(payload, dict):
            if "sample_to_cluster" in payload:
                cluster_ids = _cluster_ids_from_sample_to_cluster(
                    payload["sample_to_cluster"]
                )
            elif "cluster_to_samples" in payload:
                cluster_ids = _cluster_ids_from_cluster_to_samples(
                    payload["cluster_to_samples"]
                )
            elif any(str(k).startswith("cluster_") for k in payload.keys()):
                raise ValueError(
                    "cluster_all.json only stores cluster previews and cannot be reused "
                    "as fixed assignments. Please use cluster_ids_*.npy or "
                    "cluster_assignments_*.json instead."
                )
            else:
                raise ValueError(
                    "Unsupported JSON cluster format. Expected sample_to_cluster, "
                    "cluster_to_samples, or a plain list."
                )
        else:
            raise ValueError(
                f"Unsupported JSON payload type: {type(payload).__name__}"
            )
    else:
        raise ValueError(
            f"Unsupported precomputed cluster format: {ext}. Use .npy or .json."
        )

    cluster_ids = np.asarray(cluster_ids, dtype=np.int32).reshape(-1)
    if cluster_ids.size == 0:
        raise ValueError("Loaded precomputed cluster ids are empty.")
    if np.any(cluster_ids < 0):
        raise ValueError("Cluster ids must be non-negative.")
    return cluster_ids
