"""
ClusterPrototypes
=================
Learnable per-cluster prototype vectors μ_k used by Joint Loss Level 1.

Shape: [K, H]   where K = n_clusters and H = hidden_size.

The prototype tensor is stored as a single nn.Parameter and lives in a
dedicated optimizer parameter group (usually with a higher lr and
weight_decay=0). The forward() simply gathers μ_{k_i} for a batch of
cluster ids.

This module is only active when cfg.joint_loss.enabled == true; it is
written so that importing it has zero side effect on legacy runs.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

logger = logging.getLogger(__name__)


# ======================================================================
# Prototype module
# ======================================================================


class ClusterPrototypes(nn.Module):
    """Learnable cluster prototypes μ ∈ R^{K×H}.

    Args:
        n_clusters: K
        hidden_size: H
        init_tensor: optional [K, H] float tensor used to initialise μ.
                     If None, uses N(0, 1/sqrt(H)).
        distance:    "cosine" | "squared_l2"
        dtype:       storage dtype for μ (prototype is held in fp32 by
                     default for stable optimisation; the forward casts
                     to the hidden dtype on-the-fly if needed).
    """

    VALID_DISTANCES = ("cosine", "squared_l2")

    def __init__(
        self,
        n_clusters: int,
        hidden_size: int,
        init_tensor: Optional[torch.Tensor] = None,
        distance: str = "cosine",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if distance not in self.VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {self.VALID_DISTANCES}, got {distance!r}"
            )

        self.n_clusters = int(n_clusters)
        self.hidden_size = int(hidden_size)
        self.distance = distance

        if init_tensor is None:
            init_tensor = torch.randn(n_clusters, hidden_size, dtype=dtype) / (
                hidden_size ** 0.5
            )
        else:
            if init_tensor.shape != (n_clusters, hidden_size):
                raise ValueError(
                    f"init_tensor shape {tuple(init_tensor.shape)} != "
                    f"({n_clusters}, {hidden_size})"
                )
            init_tensor = init_tensor.detach().to(dtype=dtype).clone()

        self.mu = nn.Parameter(init_tensor, requires_grad=True)

        # Cached copy of μ at last log call, used to compute ‖Δμ‖ between
        # successive prototype_history entries. Not a buffer to avoid
        # checkpoint bloat.
        self._mu_last_logged: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward / distance
    # ------------------------------------------------------------------

    def forward(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        """Gather μ for a batch of cluster ids.

        Args:
            cluster_ids: LongTensor [B] with values in [0, K).

        Returns:
            [B, H] tensor with the same dtype as self.mu.
        """
        if cluster_ids.dtype not in (torch.int32, torch.int64):
            cluster_ids = cluster_ids.long()
        else:
            cluster_ids = cluster_ids.long()
        return self.mu[cluster_ids]

    def pull_distance(
        self,
        hidden: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample pull distance dist(h_i, μ_{k_i}).

        Args:
            hidden:      [B, H] sample hidden (same device as μ).
            cluster_ids: [B].

        Returns:
            [B] float tensor, always in the dtype of `hidden`.
        """
        mu_b = self.forward(cluster_ids).to(dtype=hidden.dtype)

        if self.distance == "cosine":
            # 1 - cos(h, μ). eps chosen so bf16 norms don't explode.
            h_norm = hidden / hidden.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            mu_norm = mu_b / mu_b.norm(dim=-1, keepdim=True).clamp(min=1e-4)
            cos = (h_norm * mu_norm).sum(dim=-1)
            return 1.0 - cos

        # squared_l2
        diff = hidden - mu_b
        return diff.pow(2).sum(dim=-1) / self.hidden_size

    # ------------------------------------------------------------------
    # Monitoring (collapse / drift)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def pairwise_metrics(self) -> Dict[str, float]:
        """Compute scalar metrics for monitoring (see design §11)."""
        mu = self.mu.detach().float()  # [K, H]
        k = mu.shape[0]
        # Frobenius norm and per-row mean norm
        frob = float(mu.norm().item())
        row_norms = mu.norm(dim=-1)
        mean_row_norm = float(row_norms.mean().item())

        # Pairwise cosine on the upper triangle (exclude diagonal).
        mu_n = mu / row_norms.clamp(min=1e-8).unsqueeze(-1)
        cos_mat = mu_n @ mu_n.t()  # [K, K]
        if k >= 2:
            triu = torch.triu_indices(k, k, offset=1)
            pair = cos_mat[triu[0], triu[1]]
            pc_min = float(pair.min().item())
            pc_max = float(pair.max().item())
            pc_mean = float(pair.mean().item())
        else:
            pc_min = pc_max = pc_mean = 0.0

        # Step delta since last logged snapshot.
        if self._mu_last_logged is not None:
            delta = (mu - self._mu_last_logged.to(mu.device)).norm()
            delta_norm = float(delta.item())
        else:
            delta_norm = float("nan")
        self._mu_last_logged = mu.cpu()

        return {
            "proto_frob": frob,
            "proto_row_norm_mean": mean_row_norm,
            "pairwise_cos_min": pc_min,
            "pairwise_cos_mean": pc_mean,
            "pairwise_cos_max": pc_max,
            "step_delta_norm": delta_norm,
        }


# ======================================================================
# Prototype initialisation (per-cluster hidden mean)
# ======================================================================


@torch.no_grad()
def build_prototype_init_from_model(
    raw_model: nn.Module,
    train_dataset,
    cluster_ids: np.ndarray,
    n_clusters: int,
    layer_idx: int,
    hidden_size: int,
    device: torch.device,
    batch_size: int = 8,
    dtype_accum: torch.dtype = torch.float32,
    max_samples_per_cluster: int = -1,
    verbose: bool = True,
) -> Tuple[torch.Tensor, np.ndarray, float]:
    """Extract [K, H] prototype init via per-cluster masked-mean pooling
    of layer ``layer_idx`` hidden states using the training model itself.

    We iterate ``train_dataset`` in its natural order exactly once. For
    each batch we run `raw_model(output_hidden_states=True)` and then
    accumulate ``hidden[layer_idx+1]`` into per-cluster running sums
    (fp32 accumulators).

    Args:
        raw_model:    the (non-DDP, non-DeepSpeed) HF CausalLM with the
                      training model's weights already loaded on ``device``.
                      The model is put in eval mode during extraction and
                      restored to train mode at the end.
        train_dataset: expected to have ``.collate(list_of_items)`` and
                      to yield ``(index, data)`` from ``__getitem__``.
                      This matches ``JsonFolderDataset``.
        cluster_ids:  np.ndarray [N] mapping each sample index to a
                      cluster id in [0, K).
        n_clusters:   K.
        layer_idx:    0-based transformer layer index. ``layer_idx+1`` is
                      used when indexing into ``outputs.hidden_states``
                      (index 0 is the embedding output).
        hidden_size:  H, for sanity check.
        device:       cuda device to run the forward on.
        batch_size:   mini-batch used for extraction.
        dtype_accum:  accumulator dtype (default fp32). We do bf16 forward
                      and cast to fp32 before summing.
        max_samples_per_cluster: if > 0, stop accumulating for a cluster
                      after it has seen this many samples. -1 = no cap.
        verbose:      log progress every 50 batches.

    Returns:
        init:        [K, H] float32 tensor (CPU) with per-cluster
                     hidden means. Empty clusters get a zero row + warning.
        counts:      [K] int64 np.ndarray of how many samples contributed.
        duration_s:  wall-clock seconds.
    """
    if cluster_ids.shape[0] != len(train_dataset):
        raise ValueError(
            f"cluster_ids length {cluster_ids.shape[0]} != dataset length "
            f"{len(train_dataset)}"
        )
    cluster_ids_t = torch.from_numpy(
        np.asarray(cluster_ids, dtype=np.int64)
    )  # kept on CPU, indexed per-batch

    sums = torch.zeros(n_clusters, hidden_size, dtype=dtype_accum)  # CPU
    counts = torch.zeros(n_clusters, dtype=torch.int64)  # CPU

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(train_dataset),
        collate_fn=train_dataset.collate,
        num_workers=0,
        drop_last=False,
    )

    # Put model in eval mode during extraction, restore afterwards.
    was_training = raw_model.training
    raw_model.eval()

    start = time.time()
    running_sample_idx = 0  # dataset-wise pointer (SequentialSampler)
    full_clusters_cpu = None  # only meaningful when max_samples_per_cluster > 0

    try:
        for batch_idx, (model_batch, no_model_batch) in enumerate(loader):
            if model_batch is None:
                continue
            bsz = model_batch["input_ids"].shape[0]
            ids_cpu = cluster_ids_t[running_sample_idx : running_sample_idx + bsz]
            running_sample_idx += bsz

            # Optional early-skip: if every cluster represented in this
            # batch has already hit the cap, skip the forward.
            if max_samples_per_cluster > 0:
                full_clusters_cpu = counts >= max_samples_per_cluster
                active = ~full_clusters_cpu[ids_cpu]
                if not active.any():
                    continue
            else:
                active = torch.ones(bsz, dtype=torch.bool)

            # Move to device
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)

            outputs = raw_model(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors.
            # Index 0 = embedding output, index i+1 = layer i output.
            h = outputs.hidden_states[layer_idx + 1]  # [B, L, H]

            mask = model_batch["attention_mask"].to(h.dtype).unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, H]
            pooled = pooled.to(dtype=dtype_accum, device="cpu")

            # Gather valid rows (under the cap) and accumulate.
            active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if active_idx.numel() > 0:
                pooled_a = pooled[active_idx]
                ids_a = ids_cpu[active_idx]
                sums.index_add_(0, ids_a, pooled_a)
                counts.index_add_(
                    0,
                    ids_a,
                    torch.ones(ids_a.numel(), dtype=torch.int64),
                )

            if verbose and (batch_idx + 1) % 50 == 0:
                coverage = int((counts > 0).sum().item())
                logger.info(
                    f"[proto-init] batch {batch_idx + 1}/{len(loader)} "
                    f"samples={running_sample_idx}/{len(train_dataset)} "
                    f"clusters_seen={coverage}/{n_clusters}"
                )

            # If we've capped every cluster, stop early.
            if (
                max_samples_per_cluster > 0
                and bool((counts >= max_samples_per_cluster).all().item())
            ):
                logger.info(
                    f"[proto-init] early stop at batch {batch_idx + 1}: every "
                    f"cluster reached cap={max_samples_per_cluster}"
                )
                break
    finally:
        if was_training:
            raw_model.train()

    duration_s = time.time() - start

    # Compute mean; guard empty clusters.
    empty = counts == 0
    n_empty = int(empty.sum().item())
    if n_empty > 0:
        logger.warning(
            f"[proto-init] {n_empty}/{n_clusters} clusters had zero "
            f"samples contributing; their prototypes are initialised to zero."
        )
    safe_counts = counts.clamp(min=1).to(sums.dtype).unsqueeze(-1)
    init = sums / safe_counts
    init[empty] = 0.0

    return init.contiguous(), counts.numpy().astype(np.int64), duration_s


# ======================================================================
# Utility → per-cluster pull weight
# ======================================================================


def build_utility_pull_weights(
    csv_path: str,
    n_clusters: int,
    abilities,
    weights,
    aggregator: str,
    normalize: str,
    sign: str,
    floor: float = 0.0,
    base_to_meta_csv: Optional[str] = None,
) -> np.ndarray:
    """Map an Exp2 utility CSV to a non-negative per-cluster pull weight
    vector w ∈ R^K with mean(w) ≈ 1.

    We intentionally reuse ``analysis.utility_prior.load_utility_prior``
    to aggregate / normalise / sign the CSV, then convert the signed
    "prior" into a strictly-non-negative weight via:

        raw = sign * normalized_utility       (from load_utility_prior with alpha=1)
        w   = clip(raw, min=floor) + small_eps
        w   = w * n_clusters / w.sum()        (so mean(w) = 1)

    Rationale: the upstream helper emits grad_gamma-style offsets which
    can be negative (low-utility cluster ⇒ negative prior). For a pull
    WEIGHT we need w_k ≥ 0, and we want E[w] = 1 so that the global
    strength is comparable to the "uniform w_k = 1" baseline regardless
    of utility scale.

    Args:
        csv_path:    path to a utility_matrix_*.csv aligned with
                     cluster_id 0..K-1.
        n_clusters:  expected K.
        abilities, weights, aggregator, normalize, sign: forwarded to
                     load_utility_prior. Use sign="positive" to have
                     high-utility clusters receive larger w_k.
        floor:       minimum per-cluster weight floor (before renorm).
                     Set >0 if you want low-utility clusters to still
                     receive some pull.
        base_to_meta_csv: optional path to base_cluster_to_meta_cluster.csv
                     (columns: base_cluster_id, meta_cluster_id). When given,
                     the utility vector is aggregated from K base clusters
                     to M meta-clusters before renorm.

    Returns:
        np.ndarray [K], float32, non-negative, mean ≈ 1.
    """
    # Local import so this file is importable even if analysis/ is not on
    # PYTHONPATH yet (trainer is imported before analysis in some flows).
    from analysis.utility_prior import load_utility_prior

    # Determine the number of base clusters in the CSV.
    # If base_to_meta_csv is provided, the CSV has K_base clusters,
    # and we need to call load_utility_prior with K_base, then aggregate.
    if base_to_meta_csv is not None:
        import csv as _csv_tmp
        _base_ids = set()
        with open(base_to_meta_csv, "r", encoding="utf-8") as _f:
            _reader = _csv_tmp.DictReader(_f)
            for _row in _reader:
                _base_ids.add(int(_row["base_cluster_id"]))
        _n_base = max(_base_ids) + 1  # assume 0-indexed
    else:
        _n_base = n_clusters

    # Call with alpha=1 and the requested sign — we'll interpret the
    # returned vector ourselves. load_utility_prior returns:
    #   sign=positive:  init = -1 * normalized_utility   (centered)
    #   sign=negative:  init = +1 * normalized_utility   (centered)
    # Undo the sign flip so `raw` is aligned with utility direction the
    # caller asked for: larger raw ⇒ higher utility.
    init = load_utility_prior(
        csv_path=csv_path,
        n_clusters=_n_base,  # use base cluster count, not meta count
        abilities=list(abilities or []),
        weights=list(weights or []),
        aggregator=aggregator,
        normalize=normalize,
        alpha=1.0,
        sign=sign,
    )
    # load_utility_prior returns zero when alpha=0; here alpha=1, so init
    # carries sign info. Recover "higher ⇒ more utility":
    if str(sign).lower() == "positive":
        raw = -np.asarray(init, dtype=np.float64)  # undo the flip
    else:
        raw = np.asarray(init, dtype=np.float64)

    # --- optional: aggregate base → meta ---
    if base_to_meta_csv is not None:
        import csv as _csv2
        _meta_of_base = {}
        with open(base_to_meta_csv, "r", encoding="utf-8") as _f:
            _reader = _csv2.DictReader(_f)
            for _row in _reader:
                _b = int(_row["base_cluster_id"])
                _m = int(_row["meta_cluster_id"])
                _meta_of_base[_b] = _m
        _M = max(_meta_of_base.values()) + 1
        _agg = np.zeros(_M, dtype=np.float64)
        for _b, _m in _meta_of_base.items():
            if _b < len(raw):
                _agg[_m] += raw[_b]
        raw = _agg  # now length M

    w = np.clip(raw, a_min=float(floor), a_max=None) + 1e-8
    total = float(w.sum())
    if total <= 0.0:
        logger.warning(
            "[utility-pull-weights] all weights non-positive after flooring; "
            "falling back to uniform w_k = 1."
        )
        return np.ones(n_clusters, dtype=np.float32)

    w = w * (float(n_clusters) / total)  # E[w_k] = 1
    return w.astype(np.float32)


# ======================================================================
# I/O helpers
# ======================================================================


def save_prototype_init(
    save_dir: str,
    mu_init: torch.Tensor,
    counts: np.ndarray,
    layer_idx: int,
    duration_s: float,
    extra: Optional[dict] = None,
) -> None:
    """Persist μ^{(0)} and its metadata under ``save_dir``."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "prototype_init.npy", mu_init.detach().cpu().numpy())
    meta = {
        "layer_idx": int(layer_idx),
        "n_clusters": int(mu_init.shape[0]),
        "hidden_size": int(mu_init.shape[1]),
        "samples_per_cluster": counts.tolist(),
        "n_empty_clusters": int((counts == 0).sum()),
        "duration_s": float(duration_s),
    }
    if extra:
        meta.update(extra)
    with (path / "prototype_init_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_prototype_final(save_dir: str, mu_final: torch.Tensor) -> None:
    path = Path(save_dir) / "prototype_final.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mu_final.detach().cpu().numpy())


def append_prototype_history(
    save_dir: str,
    step: int,
    metrics: Dict[str, float],
    extra: Optional[dict] = None,
) -> None:
    """Append a JSONL line to ``prototype_history.jsonl``."""
    path = Path(save_dir) / "prototype_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": int(step), **metrics}
    if extra:
        record.update(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
