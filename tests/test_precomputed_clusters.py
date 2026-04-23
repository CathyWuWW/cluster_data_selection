from __future__ import annotations

import json

import numpy as np
import pytest

from utils.cluster_io import load_precomputed_cluster_ids


def test_load_precomputed_cluster_ids_from_npy(tmp_path):
    path = tmp_path / "cluster_ids.npy"
    np.save(path, np.array([2, 0, 1, 2], dtype=np.int32))

    loaded = load_precomputed_cluster_ids(str(path))

    assert loaded.dtype == np.int32
    assert loaded.tolist() == [2, 0, 1, 2]


def test_load_precomputed_cluster_ids_from_sample_to_cluster_json(tmp_path):
    path = tmp_path / "cluster_assignments.json"
    path.write_text(
        json.dumps({"sample_to_cluster": {"0": 1, "1": 0, "2": 1}}),
        encoding="utf-8",
    )

    loaded = load_precomputed_cluster_ids(str(path))

    assert loaded.tolist() == [1, 0, 1]


def test_load_precomputed_cluster_ids_from_cluster_to_samples_json(tmp_path):
    path = tmp_path / "cluster_assignments.json"
    path.write_text(
        json.dumps({"cluster_to_samples": {"3": [0, 2], "1": [1]}}),
        encoding="utf-8",
    )

    loaded = load_precomputed_cluster_ids(str(path))

    assert loaded.tolist() == [3, 1, 3]


def test_cluster_all_preview_json_is_rejected(tmp_path):
    path = tmp_path / "cluster_all.json"
    path.write_text(
        json.dumps({"cluster_0": {"size": 10, "samples": [{"index": 0}]}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="cluster_all.json only stores cluster previews"
    ):
        load_precomputed_cluster_ids(str(path))


def test_non_contiguous_sample_ids_are_rejected(tmp_path):
    path = tmp_path / "cluster_assignments.json"
    path.write_text(
        json.dumps({"sample_to_cluster": {"0": 1, "2": 0}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous range starting at 0"):
        load_precomputed_cluster_ids(str(path))
