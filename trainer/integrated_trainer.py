"""
IntegratedClusterTrainer
========================
Integrates continual pre-training with on-the-fly cluster weight updates
via a simplified PMP backward pass (Hessian = 0).

Pipeline per training step:
  1. Draw a cluster-weighted batch
  2. Normal LM forward + backward + optimizer step
  3. Push (pre-step params, batch, cluster_ids) to ring buffer
  4. Every `pmp.update_interval` steps → run PMP backward on ring buffer
     → accumulate grad_gamma → update ClusterWeightedSampler weights

PMP backward (Hessian=0):
  λ_T = ∇L_dev(θ_T)
  for t = T-1 … 0:
      λ_t = ∇L_dev(θ_t) + λ_{t+1}          # no Hessian term
      ct_k += lr × mean_{n∈C_k} JVP(loss_n, λ_{t+1})
  grad_gamma += ct

Distributed: torchrun --nproc_per_node=N train.py --config ...
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

try:
    import deepspeed
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False

from clustering import build_clusterer
from data.cluster_dataset import ClusterDataset, ClusterWeightedSampler
from data.json_dataset import JsonFolderDataset
from data.eval_dataset import FewShotEvalDataset
from pmp.grad_utils import (
    compute_cluster_contributions,
    compute_cluster_contributions_ghost_ip,
    compute_dev_grad,
    compute_dev_grad_multi_domain,
)
from pmp.model_wrapper import TransformerWrapper
from trainer.ring_buffer import RingBuffer
from trainer.cluster_prototypes import (
    ClusterPrototypes,
    append_prototype_history,
    build_prototype_init_from_model,
    build_utility_pull_weights,
    save_prototype_final,
    save_prototype_init,
)
from utils.cluster_io import load_precomputed_cluster_ids

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================

def _print_rank0(msg: str, rank: int = 0):
    if rank == 0:
        logger.info(msg)


def _save_rank0(msg: str, path: str, rank: int = 0):
    if rank == 0:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _build_lr_scheduler(optimizer, cfg, total_steps: int):
    sched = cfg.training.scheduler
    warmup = cfg.training.warmup_iters
    if sched == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    elif sched == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )
    elif sched == "noam":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
            power=0.5,
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched}")


def _build_optimizer(model: nn.Module, cfg):
    lr = cfg.training.lr
    wd = cfg.training.weight_decay
    b1, b2 = cfg.training.adam_beta1, cfg.training.adam_beta2
    eps = cfg.training.adam_eps
    opt = cfg.training.optimizer.lower()
    if opt == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2), eps=eps)
    elif opt == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2), eps=eps)
    elif opt == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def _batch_to_device(batch: Dict, device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ======================================================================
# Multi-domain validation set manager
# ======================================================================

class DevDomainManager:
    """
    Manages multiple named validation domains, each with an independent weight.

    Used for:
      1. Weighted PMP dev gradient:
             ∇L_dev_weighted = Σ_d (w_d / Σw) · ∇L_dev_d(θ)
      2. Per-domain + weighted evaluation metrics.

    Domains can be added or removed at runtime, allowing the validation
    objective to evolve during training (e.g., increase weight of a target
    domain after a warm-up phase).

    Args:
        None — domains are registered via add_domain().
    """

    def __init__(self):
        # Ordered dict: domain_name → {weight: float, batches: list}
        self._domains: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Domain lifecycle
    # ------------------------------------------------------------------

    def add_domain(
        self,
        name: str,
        weight: float,
        batches_cpu: List[Tuple[Dict, Dict]],
    ):
        """
        Register (or overwrite) a validation domain.

        Args:
            name:       Unique identifier for the domain (e.g., "math", "code").
            weight:     Relative importance weight (unnormalised).
            batches_cpu:Pre-tokenised batches on CPU, as returned by
                        IntegratedClusterTrainer._cache_dev_batches().
        """
        self._domains[name] = {"weight": float(weight), "batches": batches_cpu}
        logger.info(
            f"DevDomainManager: registered domain '{name}' "
            f"(weight={weight:.4f}, n_batches={len(batches_cpu)})"
        )

    def remove_domain(self, name: str):
        """Deregister a domain.  No-op if name not present."""
        if name in self._domains:
            del self._domains[name]
            logger.info(f"DevDomainManager: removed domain '{name}'")
        else:
            logger.warning(f"DevDomainManager: domain '{name}' not found, skipping remove.")

    def update_weight(self, name: str, new_weight: float):
        """Dynamically change the weight of an existing domain."""
        if name not in self._domains:
            raise KeyError(f"DevDomainManager: domain '{name}' not registered.")
        self._domains[name]["weight"] = float(new_weight)
        logger.info(f"DevDomainManager: updated domain '{name}' weight → {new_weight:.4f}")

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------

    @property
    def domain_names(self) -> List[str]:
        return list(self._domains.keys())

    @property
    def total_weight(self) -> float:
        return sum(d["weight"] for d in self._domains.values())

    def get_domain_batches_for_pmp(
        self,
    ) -> List[Tuple[str, float, List[Tuple[Dict, Dict]]]]:
        """
        Return [(name, weight, batches_cpu)] for all registered domains.
        Used by compute_dev_grad_multi_domain().
        """
        return [
            (name, d["weight"], d["batches"])
            for name, d in self._domains.items()
        ]

    def get_domain_batches_on_device(
        self,
        device: torch.device,
    ) -> List[Tuple[str, float, List[Tuple[Dict, Dict]]]]:
        """Return domain batches moved to `device`."""
        return [
            (
                name,
                d["weight"],
                [
                    (_batch_to_device(mb, device), _batch_to_device(nmb, device))
                    for mb, nmb in d["batches"]
                ],
            )
            for name, d in self._domains.items()
        ]

    def __len__(self) -> int:
        return len(self._domains)

    def __repr__(self) -> str:
        parts = [f"{n}(w={d['weight']:.3f})" for n, d in self._domains.items()]
        return f"DevDomainManager([{', '.join(parts)}])"


# ======================================================================
# Main Trainer
# ======================================================================

class IntegratedClusterTrainer:
    """
    Continual pre-training with cluster-level data selection via PMP.

    Args:
        cfg: OmegaConf config object (loaded from YAML + CLI overrides).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rank = _rank()
        self.world_size = _world_size()
        self.device = torch.device(
            f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
            if torch.cuda.is_available()
            else "cpu"
        )

        # DeepSpeed flag
        self.use_deepspeed = getattr(getattr(cfg, "deepspeed", None), "enabled", False)
        if self.use_deepspeed and not _DEEPSPEED_AVAILABLE:
            raise ImportError(
                "DeepSpeed is enabled in config but not installed. "
                "Install with: pip install deepspeed"
            )

        # Dtype
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self.dtype = dtype_map[cfg.model.dtype if not cfg.training.fp32 else "float32"]

        self._setup_seed()

        # ---- Clustering FIRST (before loading training model) ----
        # Clustering uses a separate small model (Qwen2.5-0.5B) with its own tokenizer.
        # Run while GPU is empty, then release the small model.
        _print_rank0(f"Running {cfg.clustering.method} clustering ...", self.rank)
        cluster_ids = self._run_clustering(cfg)
        n_samples = len(cluster_ids)
        _print_rank0(f"Clustering done: {len(set(cluster_ids))} clusters from {n_samples} samples", self.rank)

        # ---- Save ALL clusters to JSON for inspection (rank 0 only) ----
        if self.rank == 0:
            from data.json_dataset import load_texts_from_dir
            from collections import defaultdict
            raw_texts = load_texts_from_dir(cfg.data.train_dir, cfg.data.text_field)
            cluster_groups = defaultdict(list)
            for idx, cid in enumerate(cluster_ids):
                cluster_groups[int(cid)].append(idx)
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(cluster_groups.items(), key=lambda x: -len(x[1]))
            all_clusters = {}
            for cid, indices in sorted_clusters:
                samples = []
                for i in indices[:3]:  # 每个cluster展示前3条
                    text = raw_texts[i] if i < len(raw_texts) else "[index out of range]"
                    samples.append({"index": i, "text": text[:300]})  # 截断300字符
                all_clusters[f"cluster_{cid}"] = {
                    "size": len(indices),
                    "samples": samples,
                }
            # Save full cluster info
            save_path = os.path.join(cfg.training.save_dir, "cluster_all.json")
            os.makedirs(cfg.training.save_dir, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_clusters, f, ensure_ascii=False, indent=2)
            _print_rank0(f"All {len(sorted_clusters)} clusters saved to {save_path}", self.rank)
            # Save summary stats
            sizes = [len(indices) for _, indices in sorted_clusters]
            _print_rank0(
                f"Cluster stats: total={len(sorted_clusters)}, "
                f"max_size={max(sizes)}, min_size={min(sizes)}, "
                f"avg_size={sum(sizes)/len(sizes):.1f}, "
                f"median={sorted(sizes)[len(sizes)//2]}",
                self.rank,
            )

            # ---- Save full sample_id → cluster_id mapping for later analysis ----
            # Two files are written:
            #   1. cluster_assignments.json  : human-readable, keeps cluster → [sample_ids]
            #      and sample_id → cluster_id  (sample_id = stable dataset index).
            #   2. cluster_ids.npy           : compact numpy array [N] aligned with
            #      the training dataset order (fast to reload for analysis).
            self._save_cluster_assignments(
                cluster_ids=cluster_ids,
                step=0,
                tag="initial",
            )

            del raw_texts

        # ---- Training model tokenizer + model (after clustering frees GPU) ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _print_rank0(f"Loading model from {cfg.model.path} ...", self.rank)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.path,
            torch_dtype=self.dtype,
            attn_implementation=cfg.model.attn_impl,
            trust_remote_code=True,
        )

        if cfg.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # ---- DeepSpeed or DDP wrapping ----
        if self.use_deepspeed:
            self.model, self.optimizer, _, self.lr_scheduler = self._init_deepspeed(cfg)
            self._raw_model = self.model.module
            if cfg.model.gradient_checkpointing:
                self._raw_model.gradient_checkpointing_enable()
                _print_rank0("[DeepSpeed] Gradient checkpointing re-enabled on inner model", self.rank)
            _print_rank0("[DeepSpeed] Engine initialized", self.rank)
        else:
            self.model = self.model.to(self.device)
            if _is_distributed():
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
                    output_device=int(os.environ.get("LOCAL_RANK", 0)),
                    find_unused_parameters=False,
                )
            self._raw_model = (
                self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel)
                else self.model
            )

        # ---- TransformerWrapper for PMP (always uses raw model, single GPU) ----
        self.model_wrapper = TransformerWrapper(self._raw_model)

        # ---- Training data tokenization (with training model's tokenizer) ----
        _print_rank0("Loading training data ...", self.rank)
        self.train_base_dataset = JsonFolderDataset(
            data_dir=cfg.data.train_dir,
            tokenizer=self.tokenizer,
            text_field=cfg.data.text_field,
            max_length=cfg.model.max_length,
            split_name="train",
        )
        _print_rank0(f"Train dataset: {len(self.train_base_dataset)} samples", self.rank)
        if len(cluster_ids) != len(self.train_base_dataset):
            raise ValueError(
                "Loaded cluster ids do not match the tokenized training dataset size: "
                f"{len(cluster_ids)} vs {len(self.train_base_dataset)}. "
                "Check that train_dir, tokenizer/max_length, and the saved assignments "
                "come from the same dataset snapshot."
            )

        self.train_dataset = ClusterDataset(self.train_base_dataset, cluster_ids)
        self.n_clusters = self.train_dataset.n_clusters
        # Keep a copy of the initial per-sample cluster ids for downstream
        # components (e.g. joint-loss prototype initialisation) that need
        # to iterate the dataset *before* any recluster event happens.
        self._cluster_ids_initial = np.asarray(cluster_ids, dtype=np.int64)

        # ---- Dev data ----
        _print_rank0("Loading dev data ...", self.rank)
        self.dev_domain_manager = DevDomainManager()

        dev_domains_cfg = getattr(cfg.data, "dev_domains", None)
        if dev_domains_cfg:
            for domain_cfg in dev_domains_cfg:
                domain_ds = JsonFolderDataset(
                    data_dir=domain_cfg.dir,
                    tokenizer=self.tokenizer,
                    text_field=cfg.data.text_field,
                    max_length=cfg.model.max_length,
                    max_samples=cfg.data.dev_num,
                    sample_seed=getattr(cfg.data, "dev_seed", cfg.training.seed),
                    split_name=domain_cfg.name,
                )
                batches = self._cache_dev_batches(domain_ds, cfg.pmp.dev_batch_size)
                self.dev_domain_manager.add_domain(domain_cfg.name, domain_cfg.weight, batches)
                _print_rank0(
                    f"Dev domain '{domain_cfg.name}': {len(domain_ds)} samples, "
                    f"{len(batches)} batches, weight={domain_cfg.weight}",
                    self.rank,
                )
        else:
            # Legacy single-domain path
            dev_dataset = JsonFolderDataset(
                data_dir=cfg.data.dev_dir,
                tokenizer=self.tokenizer,
                text_field=cfg.data.text_field,
                max_length=cfg.model.max_length,
                max_samples=cfg.data.dev_num,
                sample_seed=getattr(cfg.data, "dev_seed", cfg.training.seed),
                split_name="dev",
            )
            _print_rank0(f"Dev dataset: {len(dev_dataset)} samples", self.rank)
            batches = self._cache_dev_batches(dev_dataset, cfg.pmp.dev_batch_size)
            self.dev_domain_manager.add_domain("default", 1.0, batches)

        # Keep a flat list for backward compatibility (e.g. legacy callers)
        self.dev_batches_cpu = [
            b for _, _, blist in self.dev_domain_manager.get_domain_batches_for_pmp()
            for b in blist
        ]
        _print_rank0(
            f"Dev domains: {self.dev_domain_manager.domain_names} "
            f"(total {len(self.dev_batches_cpu)} batches)",
            self.rank,
        )

        # ---- Few-shot / Zero-shot evaluation dataset (optional) ----
        self.eval_format = getattr(cfg.data, "eval_format", "text")
        self.fewshot_eval_dataset = None
        if self.eval_format == "fewshot":
            n_shot = getattr(cfg.data, "n_shot", 5)
            # Determine eval data dir: use dev_dir for single-domain, or first domain dir
            eval_data_dir = cfg.data.dev_dir
            _print_rank0(
                f"Loading few-shot eval data (n_shot={n_shot}) from {eval_data_dir} ...",
                self.rank,
            )
            self.fewshot_eval_dataset = FewShotEvalDataset(
                data_dir=eval_data_dir,
                tokenizer=self.tokenizer,
                n_shot=n_shot,
                text_field=cfg.data.text_field,
                max_length=cfg.model.max_length,
                max_samples=cfg.data.dev_num,
                split_name="fewshot_eval",
            )
            _print_rank0(
                f"Few-shot eval dataset: {len(self.fewshot_eval_dataset)} samples",
                self.rank,
            )

        # ---- Sampler + DataLoader ----
        self.sampler = ClusterWeightedSampler(
            dataset=self.train_dataset,
            batch_size=cfg.training.batch_size * self.world_size * cfg.training.gradient_accumulation_steps,
            temperature=cfg.pmp.temperature,
            min_weight=cfg.pmp.min_weight,
            seed=cfg.training.seed,
            rank=self.rank,
            world_size=self.world_size,
            drop_bad_clusters=getattr(cfg.pmp, "drop_bad_clusters", False),
            drop_patience=int(getattr(cfg.pmp, "drop_patience", 5)),
        )

        # Wrap with IndexInjectingDataset so each batch carries __indices__
        # which the trainer uses to look up cluster_ids per sample.
        self._index_dataset = IndexInjectingDataset(self.train_dataset)
        self.train_dataloader = DataLoader(
            self._index_dataset,
            batch_size=cfg.training.batch_size,
            sampler=self.sampler,
            collate_fn=self._index_dataset.collate,
            num_workers=cfg.data.num_workers,
            drop_last=True,
        )

        # ---- Optimizer & Scheduler ----
        self.total_steps = cfg.training.total_iters
        if not self.use_deepspeed:
            # DeepSpeed creates optimizer & scheduler internally
            self.optimizer = _build_optimizer(self.model, cfg)
            self.lr_scheduler = _build_lr_scheduler(self.optimizer, cfg, self.total_steps)

        # ---- PMP state ----
        # Check if we need param_dim (only for legacy GhostGradProjector path)
        ghost_ip_cfg = getattr(cfg.pmp, "ghost_ip", None)
        ghost_ip_enabled = ghost_ip_cfg is not None and getattr(ghost_ip_cfg, "enabled", False)
        use_count_sketch = ghost_ip_enabled and str(getattr(ghost_ip_cfg, "proj_type", "count_sketch")) == "count_sketch"

        if not use_count_sketch:
            # Legacy path needs param_dim for projection matrix / ring buffer
            if self.use_deepspeed:
                with deepspeed.zero.GatheredParameters(
                    list(self._raw_model.parameters()), modifier_rank=None
                ):
                    param_dim = sum(p.numel() for p in self._raw_model.parameters())
            else:
                param_dim = sum(p.numel() for p in self._raw_model.parameters())
        else:
            # CountSketch doesn't need param_dim — no explicit projection matrix
            param_dim = 0

        self.ring_buffer = RingBuffer(capacity=cfg.pmp.window_size, param_dim=param_dim)
        self.grad_gamma = torch.zeros(self.n_clusters, dtype=torch.float32)

        # ---- Optional Exp2 utility prior on grad_gamma (non-invasive) ----
        # Only modifies the initial value of grad_gamma. PMP updates still drive
        # subsequent dynamics; sampler / loss / gradient code stays untouched.
        prior_cfg = getattr(cfg.pmp, "utility_prior", None)
        if prior_cfg is not None and bool(getattr(prior_cfg, "enabled", False)):
            try:
                from analysis.utility_prior import load_utility_prior
                prior_vec = load_utility_prior(
                    csv_path=str(getattr(prior_cfg, "csv_path", "")),
                    n_clusters=self.n_clusters,
                    abilities=list(getattr(prior_cfg, "abilities", []) or []),
                    weights=list(getattr(prior_cfg, "weights", []) or []),
                    aggregator=str(getattr(prior_cfg, "aggregator", "sum")),
                    normalize=str(getattr(prior_cfg, "normalize", "zscore")),
                    alpha=float(getattr(prior_cfg, "alpha", 0.0)),
                    sign=str(getattr(prior_cfg, "sign", "positive")),
                )
                prior_tensor = torch.from_numpy(prior_vec).float()
                # 1) seed the accumulated grad_gamma so PMP updates start from
                #    a non-zero baseline. With sign=positive, prior_vec already
                #    encodes -alpha*u (high-utility cluster ⇒ negative gg ⇒
                #    larger softmax weight via logits = -gg/T).
                self.grad_gamma = prior_tensor.clone()
                # 2) install a persistent sampler prior_bias that survives the
                #    softmax/min_weight clamp. update_weights uses
                #        logits = -(gg - prior_bias) / T,
                #    so the equivalent persistent shift on logits is
                #    +prior_bias/T. To keep "high utility ⇒ high weight" we
                #    need +alpha*u on the logits, i.e. prior_bias = -prior_vec.
                self.sampler.set_prior(-prior_tensor)
                w0 = self.sampler.weights
                _print_rank0(
                    f"[utility_prior] enabled: csv={prior_cfg.csv_path} "
                    f"alpha={float(prior_cfg.alpha):.4f} sign={prior_cfg.sign} "
                    f"agg={prior_cfg.aggregator} norm={prior_cfg.normalize} "
                    f"grad_gamma_init: min={float(self.grad_gamma.min()):.4f} "
                    f"max={float(self.grad_gamma.max()):.4f} "
                    f"norm={float(self.grad_gamma.norm()):.4f} ; "
                    f"sampler.weights@init: min={float(w0.min()):.4f} "
                    f"max={float(w0.max()):.4f} "
                    f"std={float(w0.std()):.4f}",
                    self.rank,
                )
            except Exception as e:
                _print_rank0(
                    f"[utility_prior] FAILED to load prior, falling back to zeros: {e}",
                    self.rank,
                )
                self.grad_gamma = torch.zeros(self.n_clusters, dtype=torch.float32)

        # ---- CountSketch / Ghost IP projector (optional fast path) ----
        self.ghost_ip_projector = None
        self.count_sketch_projector = None
        if ghost_ip_enabled:
            proj_type = str(getattr(ghost_ip_cfg, "proj_type", "count_sketch"))
            if proj_type == "count_sketch":
                from pmp.count_sketch import CountSketchProjector
                sketch_seed = int(getattr(ghost_ip_cfg, "seed", cfg.training.seed))
                self.count_sketch_projector = CountSketchProjector(
                    sketch_dim=int(ghost_ip_cfg.proj_dim),
                    seed=sketch_seed,
                )
                _print_rank0(
                    f"CountSketch projector: sketch_dim={ghost_ip_cfg.proj_dim}, "
                    f"seed={sketch_seed}  (no projection matrix, ~60MB cache)",
                    self.rank,
                )
            else:
                # Legacy GhostGradProjector (rademacher / gaussian)
                from pmp.projection import GhostGradProjector
                self.ghost_ip_projector = GhostGradProjector(
                    param_dim=param_dim,
                    proj_dim=int(ghost_ip_cfg.proj_dim),
                    proj_type=proj_type,
                    seed=cfg.training.seed,
                    device=self.device,
                    ghost_strategy=str(ghost_ip_cfg.strategy),
                    ghost_fraction=float(ghost_ip_cfg.fraction),
                )
                _print_rank0(
                    f"Ghost IP projector: proj_dim={ghost_ip_cfg.proj_dim}, "
                    f"strategy={ghost_ip_cfg.strategy}, fraction={ghost_ip_cfg.fraction}",
                    self.rank,
                )

        # ---- Optional proxy dataset ----
        self.proxy_dataset = None
        if cfg.proxy.proxy_dir:
            _print_rank0(f"Loading proxy data from {cfg.proxy.proxy_dir} ...", self.rank)
            self.proxy_dataset = JsonFolderDataset(
                data_dir=cfg.proxy.proxy_dir,
                tokenizer=self.tokenizer,
                text_field=cfg.data.text_field,
                max_length=cfg.model.max_length,
                max_samples=cfg.proxy.proxy_num,
                split_name="proxy",
            )

        # ---- Logging setup ----
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        from datetime import datetime
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(cfg.training.save_dir, f"{log_timestamp}.log")
        if self.rank == 0:
            _save_rank0(f"Config:\n{OmegaConf.to_yaml(cfg)}", self.log_file)

        # ---- Joint Clustering Loss — Level 1 (optional) ----
        # See docs/joint_loss_level1_design.md. When disabled, this call
        # is a no-op and no prototype/utility state is created, so the
        # legacy LM-only path stays byte-identical.
        self.cluster_prototypes: Optional[ClusterPrototypes] = None
        self.joint_pull_weights: Optional[torch.Tensor] = None  # [K] on self.device
        self._joint_cfg = getattr(cfg, "joint_loss", None)
        self._joint_enabled = bool(
            self._joint_cfg is not None and getattr(self._joint_cfg, "enabled", False)
        )
        if self._joint_enabled:
            self._init_joint_loss()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_seed(self):
        seed = self.cfg.training.seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_deepspeed(self, cfg):
        """
        Initialise DeepSpeed engine with ZeRO-3 config.

        Reads the JSON config file specified by cfg.deepspeed.config_file,
        patches "auto" fields with values from the training config, then
        calls deepspeed.initialize().

        Returns:
            (engine, optimizer, _, lr_scheduler)
        """
        ds_config_file = getattr(cfg.deepspeed, "config_file", "configs/ds_zero3.json")
        _print_rank0(f"[DeepSpeed] Loading config from {ds_config_file}", self.rank)

        with open(ds_config_file, "r") as f:
            ds_config = json.load(f)

        # Patch "auto" fields from our training config
        if ds_config.get("train_micro_batch_size_per_gpu") == "auto":
            ds_config["train_micro_batch_size_per_gpu"] = cfg.training.batch_size
        if ds_config.get("gradient_accumulation_steps") == "auto":
            ds_config["gradient_accumulation_steps"] = cfg.training.gradient_accumulation_steps
        if ds_config.get("train_batch_size") == "auto":
            ds_config["train_batch_size"] = (
                cfg.training.batch_size
                * cfg.training.gradient_accumulation_steps
                * self.world_size
            )
        if ds_config.get("gradient_clipping") == "auto":
            ds_config["gradient_clipping"] = cfg.training.clip_grad

        # Patch ZeRO "auto" fields
        zero_cfg = ds_config.get("zero_optimization", {})
        for key in ["reduce_bucket_size", "stage3_prefetch_bucket_size", "stage3_param_persist_threshold"]:
            if zero_cfg.get(key) == "auto":
                zero_cfg[key] = 5e8  # reasonable default ~500M elements

        # Build optimizer config for DeepSpeed
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": cfg.training.lr,
                "betas": [cfg.training.adam_beta1, cfg.training.adam_beta2],
                "eps": cfg.training.adam_eps,
                "weight_decay": cfg.training.weight_decay,
            },
        }

        # Build scheduler config for DeepSpeed
        if cfg.training.scheduler == "cosine":
            lr_min = getattr(cfg.training, "lr_min", 0.0)
            lr_max = cfg.training.lr
            ds_config["scheduler"] = {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_ratio": lr_min / lr_max if lr_max > 0 else 0.0,
                    "warmup_num_steps": cfg.training.warmup_iters,
                    "cos_min_ratio": lr_min / lr_max if lr_max > 0 else 0.0001,
                    "total_num_steps": cfg.training.total_iters,
                },
            }
        elif cfg.training.scheduler == "constant":
            ds_config["scheduler"] = {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": cfg.training.lr,
                    "warmup_num_steps": cfg.training.warmup_iters,
                },
            }
        else:
            # For noam / other schedulers, fall back to WarmupDecayLR
            ds_config["scheduler"] = {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0.0,
                    "warmup_max_lr": cfg.training.lr,
                    "warmup_num_steps": cfg.training.warmup_iters,
                    "total_num_steps": cfg.training.total_iters,
                },
            }

        _print_rank0(f"[DeepSpeed] Effective config: {json.dumps(ds_config, indent=2)}", self.rank)

        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )
        return engine, optimizer, _, lr_scheduler

    def _cache_dev_batches(
        self, dev_dataset: JsonFolderDataset, dev_batch_size: int
    ) -> List[Tuple[Dict, Dict]]:
        """Pre-tokenise dev set and cache as list of (model_batch, no_model_batch) on CPU."""
        loader = DataLoader(
            dev_dataset,
            batch_size=dev_batch_size,
            shuffle=False,
            collate_fn=dev_dataset.collate,
            num_workers=0,
            drop_last=False,
        )
        cached = []
        for model_batch, no_model_batch in loader:
            if model_batch is None:
                continue
            cached.append((model_batch, no_model_batch))
        return cached

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        rank = self.rank
        device = self.device
        gacc = cfg.training.gradient_accumulation_steps
        clip_grad = cfg.training.clip_grad

        self.model.train()
        global_step = 0
        micro_step = 0
        train_iter = iter(self.train_dataloader)

        pbar = tqdm(
            total=self.total_steps,
            desc="Training",
            disable=(rank != 0),
        )

        # Initial eval
        if not cfg.training.no_eval_at_start:
            eval_results = self._evaluate_multi_domain()
            domain_str = "  ".join(f"{k}={v:.4f}" for k, v in eval_results.items())
            self._log(f"[init] eval: {domain_str}", global_step)
            torch.cuda.empty_cache()  # free eval activations before training

        # Accumulate buffers
        accumulated_loss = 0.0
        accumulated_lm = 0.0
        accumulated_pull = 0.0
        current_batch_cluster_ids: Optional[torch.Tensor] = None
        current_combined_batch: Optional[Dict] = None

        while global_step < self.total_steps:
            # ---- Get next micro-batch ----
            try:
                model_batch, no_model_batch = next(train_iter)
            except StopIteration:
                # New epoch
                self.sampler.set_epoch(global_step)
                train_iter = iter(self.train_dataloader)
                model_batch, no_model_batch = next(train_iter)

            if model_batch is None:
                continue

            self._index_dataset.move_to_device(model_batch, no_model_batch, device)

            # Resolve cluster ids for this micro-batch
            # __getitem__ returns (idx, data); here we track idx → cluster_id
            # We piggy-back on the index by using a tracked sampler state.
            # Instead, we retrieve cluster_ids from the batch indices.
            # The DataLoader does not expose indices by default, so we
            # store them via a custom collate wrapper. See note below.
            batch_indices = model_batch.pop("__indices__", None)
            if batch_indices is not None:
                cluster_ids_batch = torch.tensor(
                    [self.train_dataset.cluster_ids[int(i)] for i in batch_indices],
                    dtype=torch.int64,
                )
            else:
                # Fallback: assign all to cluster 0 (shouldn't happen)
                cluster_ids_batch = torch.zeros(
                    model_batch["input_ids"].shape[0], dtype=torch.int64
                )

            # ---- Save params BEFORE this update for ring buffer ----
            # CountSketch mode doesn't need params history (no ring-buffer rollback),
            # so we skip the expensive GatheredParameters + get_params_vec.
            if (micro_step % gacc) == 0:
                if self.count_sketch_projector is not None:
                    params_before_update = torch.tensor([0.0])  # dummy placeholder
                else:
                    params_before_update = self.model_wrapper.get_params_vec().cpu()

            # ---- Forward ----
            combined = {**model_batch, **no_model_batch}
            loss, loss_aux = self._compute_joint_loss(
                model_batch, no_model_batch, cluster_ids_batch=cluster_ids_batch
            )
            loss_scaled = loss / gacc

            # ---- Backward ----
            if self.use_deepspeed:
                self.model.backward(loss_scaled)
            else:
                loss_scaled.backward()
            accumulated_loss += loss.item()
            accumulated_lm += float(loss_aux["lm_loss"].item())
            if loss_aux["pull_loss"] is not None:
                accumulated_pull += float(loss_aux["pull_loss"].item())

            # Accumulate batch for ring buffer
            if current_combined_batch is None:
                current_combined_batch = {k: v.detach().cpu() for k, v in combined.items()}
                current_batch_cluster_ids = cluster_ids_batch.cpu()
            else:
                for k in current_combined_batch:
                    current_combined_batch[k] = torch.cat(
                        [current_combined_batch[k], combined[k].detach().cpu()], dim=0
                    )
                current_batch_cluster_ids = torch.cat(
                    [current_batch_cluster_ids, cluster_ids_batch.cpu()], dim=0
                )

            micro_step += 1

            if micro_step % gacc != 0:
                continue

            # ---- Gradient step ----
            if self.use_deepspeed:
                # DeepSpeed handles grad clipping, optimizer step, and zero_grad internally
                self.model.step()
            else:
                if clip_grad > 0:
                    if self.cluster_prototypes is not None:
                        clip_params = list(self.model.parameters()) + list(
                            self.cluster_prototypes.parameters()
                        )
                    else:
                        clip_params = list(self.model.parameters())
                    nn.utils.clip_grad_norm_(clip_params, clip_grad)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            global_step += 1

            # ---- Push to ring buffer ----
            # We save params BEFORE this step (captured above)
            self.ring_buffer.push(
                params_before_update,
                current_combined_batch,
                current_batch_cluster_ids,
            )

            # Reset accumulators
            current_combined_batch = None
            current_batch_cluster_ids = None
            avg_loss = accumulated_loss / gacc
            avg_lm = accumulated_lm / gacc
            avg_pull = accumulated_pull / gacc if self._joint_enabled else 0.0
            accumulated_loss = 0.0
            accumulated_lm = 0.0
            accumulated_pull = 0.0

            # ---- Logging ----
            if global_step % cfg.training.log_interval == 0:
                lr_now = self.lr_scheduler.get_last_lr()[0]
                if self._joint_enabled:
                    lam = float(self._joint_cfg.lam)
                    msg = (
                        f"step={global_step}/{self.total_steps} "
                        f"loss={avg_loss:.4f} lm={avg_lm:.4f} "
                        f"pull={avg_pull:.4f} lam*pull={lam * avg_pull:.4f} "
                        f"lr={lr_now:.2e} ring_buf={len(self.ring_buffer)}"
                    )
                else:
                    msg = (
                        f"step={global_step}/{self.total_steps} "
                        f"loss={avg_loss:.4f} lr={lr_now:.2e} "
                        f"ring_buf={len(self.ring_buffer)}"
                    )
                self._log(msg, global_step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")

            # ---- PMP Backward: update cluster weights ----
            # Note: ghost_ip fast path only needs the most recent ring-buffer entry
            # (get_latest), so > 0 is sufficient. The legacy standard path
            # still checks T < 2 internally and returns early if needed.
            if (
                global_step % cfg.pmp.update_interval == 0
                and len(self.ring_buffer) > 0
            ):
                self._run_pmp_backward_and_update(global_step)

            # ---- Prototype stats (piggyback on pmp.update_interval, or
            # joint_loss.log_interval if set to a positive value) ----
            if self._joint_enabled:
                proto_log_interval = int(getattr(self._joint_cfg, "log_interval", 0))
                if proto_log_interval > 0:
                    should_log = (global_step % proto_log_interval == 0)
                else:
                    should_log = (global_step % cfg.pmp.update_interval == 0)
                if should_log:
                    self._log_prototype_stats(global_step=global_step, event="step")

            # ---- Re-clustering ----
            if (
                cfg.clustering.recluster_interval > 0
                and global_step % cfg.clustering.recluster_interval == 0
            ):
                self._recluster(global_step)

            # ---- Evaluation ----
            if global_step % cfg.training.eval_interval == 0:
                torch.cuda.empty_cache()  # free training activations before eval
                self.model.eval()
                eval_results = self._evaluate_multi_domain()
                self.model.train()
                torch.cuda.empty_cache()  # free eval activations before resuming training
                domain_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in eval_results.items()
                )
                self._log(f"eval: {domain_str}", global_step)

            # ---- Save checkpoint ----
            if global_step % cfg.training.save_interval == 0:
                self._save_checkpoint(global_step)

            pbar.update(1)

        pbar.close()
        self._log("Training complete.", self.total_steps)
        # Persist final prototype (rank 0 only) BEFORE the final checkpoint
        # so both land under training.save_dir.
        if self._joint_enabled and self.rank == 0 and self.cluster_prototypes is not None:
            save_prototype_final(
                save_dir=self.cfg.training.save_dir,
                mu_final=self.cluster_prototypes.mu.detach().cpu(),
            )
            self._log_prototype_stats(global_step=int(self.total_steps), event="final")
        self._save_checkpoint(global_step, final=True)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _compute_lm_loss(self, model_batch: Dict, no_model_batch: Dict) -> torch.Tensor:
        """Legacy LM-only loss, used by _evaluate_multi_domain / _evaluate."""
        outputs = self.model(**model_batch, use_cache=False)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            no_model_batch["label"].view(-1),
        ).view(no_model_batch["label"].shape)
        lm_loss = (losses * no_model_batch["loss_mask"]).sum(dim=-1) / (
            no_model_batch["loss_mask"].sum(dim=-1).clamp(min=1)
        )
        return lm_loss.mean()

    def _compute_joint_loss(
        self,
        model_batch: Dict,
        no_model_batch: Dict,
        cluster_ids_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Total training loss used inside the training loop.

        When ``joint_loss.enabled=false`` this returns exactly the legacy
        LM loss and an aux dict with ``pull_loss=None``, so upstream
        accounting (gradient_accumulation scaling, logging) stays
        compatible.

        When enabled it adds the cluster-pull regulariser; see
        ``docs/joint_loss_level1_design.md``.

        Args:
            model_batch, no_model_batch: batched tensors on the training device.
            cluster_ids_batch: required iff ``self._joint_enabled``. [B] long tensor
                of meta-cluster ids for the current batch, on any device; we move
                it to ``self.device`` internally.

        Returns:
            (total_loss, aux) where aux has keys:
                "lm_loss":   detached tensor of the pre-pull LM loss.
                "pull_loss": detached pull-loss tensor (before lam multiplication),
                             or None if joint loss is disabled.
        """
        if not self._joint_enabled:
            lm = self._compute_lm_loss(model_batch, no_model_batch)
            return lm, {"lm_loss": lm.detach(), "pull_loss": None}

        # --- joint path: one forward with output_hidden_states=True ---
        outputs = self.model(
            **model_batch,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(
            logits.view(-1, logits.size(-1)),
            no_model_batch["label"].view(-1),
        ).view(no_model_batch["label"].shape)
        lm_loss = (losses * no_model_batch["loss_mask"]).sum(dim=-1) / (
            no_model_batch["loss_mask"].sum(dim=-1).clamp(min=1)
        )
        lm_loss = lm_loss.mean()

        # Pull term
        assert self.cluster_prototypes is not None, (
            "_compute_joint_loss: joint_loss enabled but prototypes not built"
        )
        if cluster_ids_batch is None:
            raise RuntimeError(
                "_compute_joint_loss: joint_loss enabled but cluster_ids_batch=None."
                " The training loop must resolve cluster ids BEFORE calling this fn."
            )

        layer_idx = int(self._joint_cfg.layer_idx)
        hidden_states = outputs.hidden_states  # tuple len = num_layers + 1
        # Validate once per run.
        max_layer = len(hidden_states) - 2  # -1 for tuple length, -1 because layer_idx is 0-based
        if layer_idx < 0 or layer_idx > max_layer:
            raise ValueError(
                f"joint_loss.layer_idx={layer_idx} out of range [0, {max_layer}] "
                f"for this model (num_layers={max_layer + 1})"
            )
        h_layer = hidden_states[layer_idx + 1]  # [B, L, H]

        mask = model_batch["attention_mask"].to(h_layer.dtype).unsqueeze(-1)
        pooled = (h_layer * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, H]

        ids = cluster_ids_batch.to(device=self.device, dtype=torch.long)
        per_sample_pull = self.cluster_prototypes.pull_distance(pooled, ids)  # [B]

        # Per-cluster pull weights w_k ∈ R^K (non-negative, mean≈1)
        w_batch = self.joint_pull_weights[ids]  # [B]
        pull_loss = (per_sample_pull * w_batch).mean()

        lam = float(self._joint_cfg.lam)
        total = lm_loss + lam * pull_loss

        return total, {
            "lm_loss": lm_loss.detach(),
            "pull_loss": pull_loss.detach(),
        }

    # ------------------------------------------------------------------
    # PMP backward pass (Hessian = 0)
    # ------------------------------------------------------------------

    def _run_pmp_backward_and_update(self, global_step: int):
        """
        Cluster-weight update via PMP backward.

        Three execution paths (selected by cfg.pmp.ghost_ip):

        ── CountSketch fast path (recommended) ──────────────────────────────
          Hash-based sketch of gradients — no explicit projection matrix.
          ct_k ≈ pmp_lr · <sketch(∇L_dev), sketch(∇L_k)>
          Memory: ~60MB cache vs 16GB for explicit projection matrix.
          ZeRO-3 native: sketch(shard) + all_reduce = sketch(full).

        ── Ghost Inner Product fast path (legacy) ───────────────────────────
          Uses explicit GhostGradProjector with random projection matrix.
          High memory (~16GB for 8.2B params).

        ── Standard ring-buffer JVP path (exact) ───────────────────────────
          Full backward traversal with per-step JVP.
        """
        cfg = self.cfg
        device = self.device
        distributed = _is_distributed()
        ws = self.world_size

        _print_rank0(f"[PMP] step={global_step}", self.rank)

        # Free training activations/gradients before PMP forward passes.
        # This is critical when GPU memory is near-full (~97GB/98GB).
        self.model.zero_grad()
        torch.cuda.empty_cache()

        # ==============================================================
        # CountSketch fast path
        # ==============================================================
        if self.count_sketch_projector is not None:
            from pmp.grad_utils_sketch import compute_cluster_contributions_sketch

            # Move ALL domain dev batches to device
            domain_batches_device = self.dev_domain_manager.get_domain_batches_on_device(device)
            flat_dev_batches = [
                b for _, _, blist in domain_batches_device for b in blist
            ]

            # Evaluate ALL clusters (each with random 4 samples from train_dataset)
            # Cluster loop is rank-sharded: each rank owns clusters[rank::world_size],
            # final all_reduce(SUM) combines contributions. ~world_size× speedup.
            grad_gamma_delta = compute_cluster_contributions_sketch(
                model=self._raw_model,
                dev_batches=flat_dev_batches,
                n_clusters=self.n_clusters,
                pmp_lr=cfg.pmp.lr,
                sketcher=self.count_sketch_projector,
                train_dataset=self.train_dataset,
                n_samples_per_cluster=4,
                world_size=ws,
                distributed=distributed,
            )

            _print_rank0(
                f"[PMP] CountSketch: grad_gamma_delta norm={grad_gamma_delta.norm():.4f}",
                self.rank,
            )

        else:
            # Legacy paths: need GatheredParameters for ZeRO-3
            if self.use_deepspeed:
                pmp_context = deepspeed.zero.GatheredParameters(
                    list(self._raw_model.parameters()), modifier_rank=None
                )
            else:
                from contextlib import nullcontext
                pmp_context = nullcontext()

            with pmp_context:
                # Save current model state so we can restore after PMP
                current_params_backup = self.model_wrapper.get_params_vec().cpu()

                # Move ALL domain dev batches to device once
                domain_batches_device = self.dev_domain_manager.get_domain_batches_on_device(device)

                # ================================================================
                # Ghost Inner Product fast path (legacy)
                # ================================================================
                ghost_ip_cfg = getattr(cfg.pmp, "ghost_ip", None)
                if (
                    ghost_ip_cfg is not None
                    and getattr(ghost_ip_cfg, "enabled", False)
                    and self.ghost_ip_projector is not None
                ):
                    latest = self.ring_buffer.get_latest()
                    if latest is None:
                        _print_rank0("[PMP] Ghost IP: ring buffer empty, skipping.", self.rank)
                        return

                    params_vec_latest, batch_cpu_latest, cluster_ids_latest = latest

                    # Use current model params (already up to date)
                    self.model_wrapper.set_params_vec(params_vec_latest.to(device))
                    params_cur = {n: p.detach() for n, p in self.model_wrapper.named_parameters()}
                    buffers_cur = {n: b.detach() for n, b in self.model_wrapper.named_buffers()}

                    # Flatten domain batches to a single list for ghost IP
                    flat_dev_batches = [
                        b for _, _, blist in domain_batches_device for b in blist
                    ]

                    batch_device = _batch_to_device(batch_cpu_latest, device)
                    cluster_ids_device = cluster_ids_latest.to(device)

                    grad_gamma_delta = compute_cluster_contributions_ghost_ip(
                        model=self.model_wrapper,
                        dev_batches=flat_dev_batches,
                        batch=batch_device,
                        batch_cluster_ids=cluster_ids_device,
                        params=params_cur,
                        buffers=buffers_cur,
                        n_clusters=self.n_clusters,
                        pmp_lr=cfg.pmp.lr,
                        ghost_projector=self.ghost_ip_projector,
                        world_size=ws,
                        distributed=distributed,
                    )

                    _print_rank0(
                        f"[PMP] Ghost IP: grad_gamma_delta norm={grad_gamma_delta.norm():.4f}",
                        self.rank,
                    )

                # ================================================================
                # Standard ring-buffer JVP path
                # ================================================================
                else:
                    history = self.ring_buffer.get_all_ordered()  # oldest → newest
                    T = len(history)
                    if T < 2:
                        return

                    _print_rank0(f"[PMP] Standard path: window_size={T}", self.rank)

                    grad_gamma_delta = torch.zeros(self.n_clusters, device=device, dtype=torch.float32)
                    lam: Optional[torch.Tensor] = None  # flat vector

                    for i in range(T - 1, -1, -1):
                        params_vec_i, batch_cpu_i, cluster_ids_i = history[i]

                        # Restore model to state at step i
                        self.model_wrapper.set_params_vec(params_vec_i.to(device))
                        params_i = {n: p.detach() for n, p in self.model_wrapper.named_parameters()}
                        buffers_i = {n: b.detach() for n, b in self.model_wrapper.named_buffers()}

                        # 1. Compute weighted multi-domain ∇L_dev(θ_i)
                        g_dev = compute_dev_grad_multi_domain(
                            self.model_wrapper,
                            domain_batches_device,
                            params_i,
                            buffers_i,
                            world_size=ws,
                            distributed=distributed,
                        )

                        # 2. Init λ at the terminal step (newest)
                        if lam is None:
                            lam = g_dev
                            continue  # no contribution at terminal step

                        # 3. Update λ (Hessian = 0 → no HVP term)
                        lam = g_dev + lam  # λ_{t-1} = ∇L_dev(θ_{t-1}) + λ_t

                        # 4. Compute cluster contributions for this step
                        lam_param = self.model_wrapper.vector_to_params(lam)
                        batch_device = _batch_to_device(batch_cpu_i, device)
                        cluster_ids_device = cluster_ids_i.to(device)

                        delta = compute_cluster_contributions(
                            model=self.model_wrapper,
                            batch=batch_device,
                            batch_cluster_ids=cluster_ids_device,
                            lam_param=lam_param,
                            params=params_i,
                            buffers=buffers_i,
                            n_clusters=self.n_clusters,
                            pmp_lr=cfg.pmp.lr,
                            chunk_size=cfg.pmp.get("jvp_chunk_size", None),
                            distributed=distributed,
                            world_size=ws,
                        )
                        grad_gamma_delta += delta

                    # All-reduce grad_gamma_delta so all ranks agree
                    if distributed:
                        dist.all_reduce(grad_gamma_delta, op=dist.ReduceOp.SUM)
                    grad_gamma_delta /= ws

            # Restore model to current training state
            self.model_wrapper.set_params_vec(current_params_backup.to(device))

        # ---- Accumulate grad_gamma and update sampler ----
        if cfg.pmp.accumulate_grad_gamma:
            self.grad_gamma = self.grad_gamma.to(device) + grad_gamma_delta
        else:
            self.grad_gamma = grad_gamma_delta.clone()

        self.sampler.update_weights(self.grad_gamma.cpu(), grad_gamma_delta.cpu())

        # ---- Append per-cluster weight history to cluster_weight_history.jsonl ----
        # One JSON line per PMP update so later analysis can reconstruct the full
        # trajectory w_k(t) for every cluster k.
        self._log_cluster_weights(
            global_step=global_step,
            grad_gamma=self.grad_gamma,
            grad_gamma_delta=grad_gamma_delta,
            event="pmp_update",
        )

        # Logging
        gg = self.grad_gamma
        _print_rank0(
            f"[PMP] grad_gamma: min={gg.min():.4f}, max={gg.max():.4f}, "
            f"norm={gg.norm():.4f}",
            self.rank,
        )
        w = self.sampler.weights
        _print_rank0(
            f"[PMP] weights: min={w.min():.4f}, max={w.max():.4f}, "
            f"entropy={(-w[w>0] * w[w>0].log()).sum():.3f}, "
            f"alive={self.sampler.n_alive}/{self.n_clusters}, "
            f"dropped={self.sampler.n_dropped}",
            self.rank,
        )

        # Save grad_gamma to disk
        if self.rank == 0:
            torch.save(
                self.grad_gamma,
                os.path.join(self.cfg.training.save_dir, f"grad_gamma_{global_step}.pt"),
            )

        # Sync all ranks before resuming DDP training
        # PMP uses _raw_model (bypasses DDP), so we must barrier here.
        if _is_distributed():
            dist.barrier()
        # Ensure model is back in train mode
        self.model.train()
        self.model.zero_grad()

    # ------------------------------------------------------------------
    # Cluster analysis artefacts (JSON logs for offline analysis)
    # ------------------------------------------------------------------

    def _save_cluster_assignments(
        self,
        cluster_ids: np.ndarray,
        step: int,
        tag: str = "initial",
    ):
        """
        Persist the sample_id → cluster_id mapping so the effect of clustering
        and (later) of cluster weights can be analysed offline.

        Outputs (all under cfg.training.save_dir):
          - cluster_ids_{tag}.npy
                compact int32 array aligned with the training dataset order.
          - cluster_assignments_{tag}.json
                {
                  "step": <int>,
                  "tag":  <str>,
                  "n_samples":  N,
                  "n_clusters": K,
                  "cluster_sizes": {cluster_id: size, ...},
                  "sample_to_cluster": {sample_id: cluster_id, ...},
                  "cluster_to_samples": {cluster_id: [sample_id, ...], ...}
                }
          - cluster_assignments_latest.json   (symlink-style convenience copy)
        """
        if self.rank != 0:
            return

        save_dir = self.cfg.training.save_dir
        os.makedirs(save_dir, exist_ok=True)

        cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
        np.save(os.path.join(save_dir, f"cluster_ids_{tag}.npy"), cluster_ids)

        # Build dictionaries. Cluster ids are ints; sample ids are dataset indices
        # into self.train_base_dataset (stable across the life of this run).
        cluster_to_samples: Dict[int, List[int]] = {}
        for sample_id, cid in enumerate(cluster_ids.tolist()):
            cluster_to_samples.setdefault(int(cid), []).append(int(sample_id))

        cluster_sizes = {cid: len(v) for cid, v in cluster_to_samples.items()}
        # sample_to_cluster may be large for huge datasets — keep it but JSON is
        # still O(N) text which is fine for ~100k-scale training sets.
        sample_to_cluster = {int(i): int(c) for i, c in enumerate(cluster_ids.tolist())}

        payload = {
            "step": int(step),
            "tag": tag,
            "n_samples": int(cluster_ids.shape[0]),
            "n_clusters": int(cluster_ids.max()) + 1 if cluster_ids.size > 0 else 0,
            "cluster_sizes": cluster_sizes,
            "sample_to_cluster": sample_to_cluster,
            "cluster_to_samples": cluster_to_samples,
        }

        out_path = os.path.join(save_dir, f"cluster_assignments_{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # Also update a "latest" pointer so analysis scripts always find the
        # most recent assignment without knowing the step number.
        latest_path = os.path.join(save_dir, "cluster_assignments_latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        _print_rank0(
            f"Cluster assignments saved: {out_path} "
            f"(N={payload['n_samples']}, K={payload['n_clusters']})",
            self.rank,
        )

    def _log_cluster_weights(
        self,
        global_step: int,
        grad_gamma: torch.Tensor,
        grad_gamma_delta: Optional[torch.Tensor] = None,
        event: str = "pmp_update",
    ):
        """
        Append one record per PMP update to cluster_weight_history.jsonl.

        The file is JSON-lines so it can be streamed without loading the whole
        history. Each record captures everything needed to reconstruct w_k(t):
            {
              "step":       <int, global optimizer step>,
              "event":      "pmp_update" | "recluster",
              "n_clusters": K,
              "weights":          [w_0, ..., w_{K-1}],   # current sampler weights
              "grad_gamma":       [g_0, ..., g_{K-1}],   # accumulated
              "grad_gamma_delta": [d_0, ..., d_{K-1}] | null,  # this update's delta
              "dead_clusters":    [bool, ...],           # True = permanently dropped
              "stats": {min, max, entropy, alive, dropped}
            }

        Cluster id k in `weights[k]` corresponds to cluster id k in the most
        recent cluster_assignments_*.json. After a "recluster" event the cluster
        ids are re-indexed, and analysis code should switch to the new
        cluster_assignments_step{N}.json.
        """
        if self.rank != 0:
            return

        save_dir = self.cfg.training.save_dir
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "cluster_weight_history.jsonl")

        gg = grad_gamma.detach().cpu().float()
        w = self.sampler.weights.detach().cpu().float()
        dead = self.sampler._dead_clusters.detach().cpu().tolist()

        w_pos = w[w > 0]
        entropy = float((-w_pos * w_pos.log()).sum().item()) if w_pos.numel() > 0 else 0.0

        record = {
            "step": int(global_step),
            "event": event,
            "n_clusters": int(w.shape[0]),
            "weights": w.tolist(),
            "grad_gamma": gg.tolist(),
            "grad_gamma_delta": (
                grad_gamma_delta.detach().cpu().float().tolist()
                if grad_gamma_delta is not None else None
            ),
            "dead_clusters": [bool(d) for d in dead],
            "stats": {
                "weight_min": float(w.min().item()),
                "weight_max": float(w.max().item()),
                "weight_entropy": entropy,
                "alive": int(self.sampler.n_alive),
                "dropped": int(self.sampler.n_dropped),
                "grad_gamma_min": float(gg.min().item()),
                "grad_gamma_max": float(gg.max().item()),
                "grad_gamma_norm": float(gg.norm().item()),
            },
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Joint Clustering Loss — Level 1 helpers
    # ------------------------------------------------------------------

    def _init_joint_loss(self) -> None:
        """Build ClusterPrototypes + pull weights + prototype optimiser group.

        Prerequisites (checked here):
          - not use_deepspeed (Level 1 only supports DDP / single-GPU).
          - self.optimizer exists (non-DeepSpeed path).
          - self.train_dataset and self._cluster_ids_initial are set.

        Side effects:
          - self.cluster_prototypes:  ClusterPrototypes on self.device.
          - self.joint_pull_weights:  [K] float tensor on self.device.
          - self.optimizer gets a new param group dedicated to prototypes
            (lr scaled by proto_lr_multiplier, weight_decay=0).
          - Writes prototype_init.npy + prototype_init_meta.json (rank 0).
        """
        cfg = self.cfg
        joint = self._joint_cfg

        if self.use_deepspeed:
            raise RuntimeError(
                "joint_loss.enabled=true is not supported with DeepSpeed in "
                "Level 1. Set deepspeed.enabled=false or joint_loss.enabled=false."
            )
        if not hasattr(self, "optimizer") or self.optimizer is None:
            raise RuntimeError(
                "_init_joint_loss: self.optimizer must exist before joint init."
            )

        # -- figure out hidden size from the raw model config --
        hf_config = getattr(self._raw_model, "config", None)
        hidden_size = int(getattr(hf_config, "hidden_size", 0))
        if hidden_size <= 0:
            raise RuntimeError(
                f"_init_joint_loss: cannot read hidden_size from {type(self._raw_model)}"
            )
        num_layers = int(getattr(hf_config, "num_hidden_layers", 0))
        layer_idx = int(joint.layer_idx)
        if not (0 <= layer_idx < num_layers):
            raise ValueError(
                f"joint_loss.layer_idx={layer_idx} out of [0, {num_layers - 1}]"
            )

        # -- build init tensor --
        init_source = str(joint.init.source)
        if init_source == "feature_mean":
            _print_rank0(
                f"[joint-loss] extracting per-cluster hidden-mean prototype init "
                f"(layer={layer_idx}, K={self.n_clusters}, H={hidden_size}, "
                f"N={len(self.train_base_dataset)})",
                self.rank,
            )
            mu_init, counts, duration_s = build_prototype_init_from_model(
                raw_model=self._raw_model,
                train_dataset=self.train_base_dataset,
                cluster_ids=self._cluster_ids_initial,
                n_clusters=self.n_clusters,
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                device=self.device,
                batch_size=int(joint.init.batch_size),
                max_samples_per_cluster=int(joint.init.max_samples_per_cluster),
                verbose=(self.rank == 0),
            )
            _print_rank0(
                f"[joint-loss] prototype init done in {duration_s:.1f}s "
                f"(empty_clusters={int((counts == 0).sum())})",
                self.rank,
            )
        elif init_source == "random":
            mu_init = None  # ClusterPrototypes default init
            counts = np.zeros(self.n_clusters, dtype=np.int64)
            duration_s = 0.0
        elif init_source == "zero":
            mu_init = torch.zeros(self.n_clusters, hidden_size, dtype=torch.float32)
            counts = np.zeros(self.n_clusters, dtype=np.int64)
            duration_s = 0.0
        else:
            raise ValueError(f"unknown joint_loss.init.source={init_source!r}")

        # -- build the module on training device --
        self.cluster_prototypes = ClusterPrototypes(
            n_clusters=self.n_clusters,
            hidden_size=hidden_size,
            init_tensor=mu_init,
            distance=str(joint.distance),
        ).to(self.device)

        # Free extraction-time activations before we start training.
        torch.cuda.empty_cache()

        # -- per-cluster pull weights --
        weight_mode = str(joint.weight.mode)
        if weight_mode == "uniform":
            w = np.ones(self.n_clusters, dtype=np.float32)
        elif weight_mode == "utility":
            csv_path = str(joint.weight.utility_csv)
            if not csv_path:
                raise ValueError(
                    "joint_loss.weight.mode=utility requires joint_loss.weight.utility_csv"
                )
            _print_rank0(
                f"[joint-loss] loading utility pull weights from {csv_path}",
                self.rank,
            )
            w = build_utility_pull_weights(
                csv_path=csv_path,
                n_clusters=self.n_clusters,
                abilities=list(getattr(joint.weight, "abilities", []) or []),
                weights=list(getattr(joint.weight, "weights", []) or []),
                aggregator=str(joint.weight.aggregator),
                normalize=str(joint.weight.normalize),
                sign=str(joint.weight.sign),
                floor=float(getattr(joint.weight, "floor", 0.0)),
                base_to_meta_csv=(
                    str(getattr(joint.weight, "base_to_meta_csv", "") or "") or None
                ),
            )
        else:
            raise ValueError(f"unknown joint_loss.weight.mode={weight_mode!r}")

        self.joint_pull_weights = torch.from_numpy(w).to(
            device=self.device, dtype=torch.float32
        )

        # -- add prototype param group to the existing optimiser --
        proto_lr = float(cfg.training.lr) * float(joint.proto_lr_multiplier)
        self.optimizer.add_param_group(
            {
                "params": list(self.cluster_prototypes.parameters()),
                "lr": proto_lr,
                "weight_decay": 0.0,
                "name": "cluster_prototypes",
            }
        )
        _print_rank0(
            f"[joint-loss] enabled: lam={float(joint.lam):.4f} "
            f"distance={str(joint.distance)} layer_idx={layer_idx} "
            f"weight_mode={weight_mode} proto_lr={proto_lr:.2e} "
            f"w_k(min/mean/max)=({float(self.joint_pull_weights.min()):.3f}/"
            f"{float(self.joint_pull_weights.mean()):.3f}/"
            f"{float(self.joint_pull_weights.max()):.3f})",
            self.rank,
        )

        # -- persist the init + meta (rank 0 only) --
        if self.rank == 0:
            save_prototype_init(
                save_dir=cfg.training.save_dir,
                mu_init=self.cluster_prototypes.mu.detach().cpu(),
                counts=counts,
                layer_idx=layer_idx,
                duration_s=duration_s,
                extra={
                    "init_source": init_source,
                    "distance": str(joint.distance),
                    "lam": float(joint.lam),
                    "weight_mode": weight_mode,
                    "weight_vector": self.joint_pull_weights.detach().cpu().tolist(),
                    "proto_lr_multiplier": float(joint.proto_lr_multiplier),
                    "hidden_size": hidden_size,
                },
            )
            # Also append a step=0 prototype_history entry so downstream
            # analysis has a non-trivial reference point.
            self._log_prototype_stats(global_step=0, event="init")

    def _log_prototype_stats(self, global_step: int, event: str = "step") -> None:
        """Append one prototype_history.jsonl record (rank 0 only)."""
        if self.rank != 0 or self.cluster_prototypes is None:
            return
        metrics = self.cluster_prototypes.pairwise_metrics()
        append_prototype_history(
            save_dir=self.cfg.training.save_dir,
            step=global_step,
            metrics=metrics,
            extra={"event": event},
        )

    # ------------------------------------------------------------------
    # Clustering (shared by __init__ and _recluster)
    # ------------------------------------------------------------------

    def _run_clustering(self, cfg) -> np.ndarray:
        """
        Run clustering and return cluster_ids for all samples.

        If cfg.clustering.embedding_model.enabled, ALL ranks load a small model
        for fast parallel feature extraction, then rank 0 runs KMeans.
        Otherwise uses the training model with ZeRO-3.
        """
        precomputed_ids_path = str(
            getattr(cfg.clustering, "precomputed_ids_path", "")
        ).strip()
        if precomputed_ids_path:
            _print_rank0(
                f"Loading precomputed cluster ids from {precomputed_ids_path} ...",
                self.rank,
            )
            cluster_ids = load_precomputed_cluster_ids(precomputed_ids_path)
            _print_rank0(
                f"Loaded {len(cluster_ids)} precomputed assignments covering "
                f"{len(set(cluster_ids.tolist()))} clusters.",
                self.rank,
            )
            return cluster_ids

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        device = self.device
        clusterer = build_clusterer(cfg)

        emb_cfg = getattr(cfg.clustering, "embedding_model", None)
        use_embed_model = emb_cfg is not None and getattr(emb_cfg, "enabled", False)
        use_random_clusterer = getattr(cfg.clustering, "method", "") == "random"

        if use_random_clusterer and not hasattr(self, "train_base_dataset"):
            # Initial random clustering runs before the training dataset/model
            # exists. Load only the tokenizer so the sample count matches the
            # later JsonFolderDataset filtering.
            _print_rank0("Preparing tokenized dataset for random clustering ...", self.rank)
            random_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.path,
                use_fast=True,
                trust_remote_code=getattr(cfg.model, "trust_remote_code", False),
            )
            if random_tokenizer.pad_token is None:
                random_tokenizer.pad_token = random_tokenizer.eos_token

            random_dataset = JsonFolderDataset(
                data_dir=cfg.data.train_dir,
                tokenizer=random_tokenizer,
                text_field=cfg.data.text_field,
                max_length=cfg.model.max_length,
                split_name="random_cluster",
            )
            cluster_ids = clusterer.fit(
                random_dataset, None, random_tokenizer,
                device, cfg, rank=self.rank, world_size=self.world_size,
            )
            del random_tokenizer, random_dataset
        elif use_embed_model:
            # ---- All ranks load the small model for parallel embedding ----
            _print_rank0(f"Loading embedding model from {emb_cfg.path} ...", self.rank)
            embed_tokenizer = AutoTokenizer.from_pretrained(
                emb_cfg.path, use_fast=True, trust_remote_code=True
            )
            if embed_tokenizer.pad_token is None:
                embed_tokenizer.pad_token = embed_tokenizer.eos_token

            embed_model = AutoModelForCausalLM.from_pretrained(
                emb_cfg.path,
                torch_dtype=dtype_map.get(getattr(emb_cfg, "dtype", "bfloat16"), torch.bfloat16),
                attn_implementation=getattr(emb_cfg, "attn_impl", "sdpa"),
                trust_remote_code=True,
            ).to(device).eval()

            # Re-tokenize training data with embedding model's tokenizer
            _print_rank0("Re-tokenizing training data for embedding model ...", self.rank)
            embed_dataset = JsonFolderDataset(
                data_dir=cfg.data.train_dir,
                tokenizer=embed_tokenizer,
                text_field=cfg.data.text_field,
                max_length=cfg.model.max_length,
                split_name="embed",
            )

            # All ranks extract features in parallel; only rank 0 runs KMeans
            cluster_ids = clusterer.fit(
                embed_dataset, embed_model, embed_tokenizer,
                device, cfg, rank=self.rank, world_size=self.world_size,
            )

            # Free embedding model memory
            del embed_model, embed_tokenizer, embed_dataset
            torch.cuda.empty_cache()
            _print_rank0("Embedding model released.", self.rank)
        else:
            # ---- Use training model (all ranks for ZeRO-3 compat) ----
            cluster_ids = clusterer.fit(
                self.train_base_dataset, self.model, self.tokenizer,
                device, cfg, rank=self.rank, world_size=self.world_size,
            )

        # Broadcast cluster_ids from rank 0 to all ranks
        if _is_distributed():
            cluster_ids_tensor = torch.tensor(cluster_ids, dtype=torch.int32).to(device)
            dist.broadcast(cluster_ids_tensor, src=0)
            cluster_ids = cluster_ids_tensor.cpu().numpy()

        return cluster_ids

    # ------------------------------------------------------------------
    # Re-clustering
    # ------------------------------------------------------------------

    def _recluster(self, global_step: int):
        _print_rank0(f"[Recluster] step={global_step}, re-running clustering ...", self.rank)
        cfg = self.cfg

        cluster_ids = self._run_clustering(cfg)

        self.train_dataset.update_cluster_ids(cluster_ids)
        n_clusters_new = self.train_dataset.n_clusters

        # ---- Persist the new sample_id → cluster_id mapping ----
        if self.rank == 0:
            self._save_cluster_assignments(
                cluster_ids=cluster_ids,
                step=global_step,
                tag=f"recluster_step{global_step}",
            )

        if n_clusters_new != self.n_clusters:
            # Resize grad_gamma if cluster count changed
            self.grad_gamma = torch.zeros(n_clusters_new, dtype=torch.float32)
            self.n_clusters = n_clusters_new

        # Reset sampler
        self.sampler = ClusterWeightedSampler(
            dataset=self.train_dataset,
            batch_size=cfg.training.batch_size * self.world_size * cfg.training.gradient_accumulation_steps,
            temperature=cfg.pmp.temperature,
            min_weight=cfg.pmp.min_weight,
            seed=cfg.training.seed + global_step,
            rank=self.rank,
            world_size=self.world_size,
            drop_bad_clusters=getattr(cfg.pmp, "drop_bad_clusters", False),
            drop_patience=int(getattr(cfg.pmp, "drop_patience", 5)),
        )
        _print_rank0(f"[Recluster] done: {n_clusters_new} clusters", self.rank)

        # Record a recluster marker in the weight history so downstream analysis
        # knows that cluster ids are no longer comparable across this boundary.
        self._log_cluster_weights(
            global_step=global_step,
            grad_gamma=self.grad_gamma,
            grad_gamma_delta=None,
            event="recluster",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        device = self.device
        loader = DataLoader(
            self.train_base_dataset,  # eval on a subset of train for simplicity
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            collate_fn=self.train_base_dataset.collate,
            num_workers=0,
            drop_last=False,
        )
        all_losses = []
        with torch.no_grad():
            for model_batch, no_model_batch in loader:
                if model_batch is None:
                    continue
                self.train_base_dataset.move_to_device(model_batch, no_model_batch, device)
                loss = self._compute_lm_loss(model_batch, no_model_batch)
                if _is_distributed():
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss /= self.world_size
                all_losses.append(loss.item())
                if len(all_losses) >= 50:  # cap eval batches
                    break
        return float(np.mean(all_losses)) if all_losses else float("inf")

    def _evaluate_multi_domain(self) -> Dict[str, float]:
        """
        Evaluate on every registered dev domain and return a result dict.

        Returns:
            Dict with one key per domain (e.g. "math", "code") containing the
            mean loss over up to 50 batches, plus a "weighted" key holding the
            normalised weighted average:
                weighted = Σ_d (w_d / Σw) · loss_d
        """
        device = self.device
        results: Dict[str, float] = {}
        total_weight = self.dev_domain_manager.total_weight

        with torch.no_grad():
            for name, d in self.dev_domain_manager._domains.items():
                domain_losses = []
                for mb_cpu, nmb_cpu in d["batches"]:
                    mb = _batch_to_device(mb_cpu, device)
                    nmb = _batch_to_device(nmb_cpu, device)
                    loss = self._compute_lm_loss(mb, nmb)
                    if _is_distributed():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss /= self.world_size
                    domain_losses.append(loss.item())
                    if len(domain_losses) >= 50:
                        break
                mean_loss = float(np.mean(domain_losses)) if domain_losses else float("inf")
                results[name] = mean_loss
                results[f"{name}_ppl"] = math.exp(mean_loss) if mean_loss < 100 else float("inf")

        # Weighted aggregate
        if total_weight > 0:
            weighted = sum(
                (d["weight"] / total_weight) * results.get(name, 0.0)
                for name, d in self.dev_domain_manager._domains.items()
            )
        else:
            weighted = float("inf")
        results["weighted"] = weighted
        results["weighted_ppl"] = math.exp(weighted) if weighted < 100 else float("inf")

        # ---- Few-shot accuracy evaluation (if enabled) ----
        if self.fewshot_eval_dataset is not None:
            fewshot_acc = self._evaluate_fewshot()
            results["fewshot_acc"] = fewshot_acc

        return results

    def _evaluate_fewshot(self) -> float:
        """Evaluate MCQ accuracy using few-shot / zero-shot prompts.

        For each prompt ending with "Answer:", we look at the logits at the
        last prompt token and compare probabilities of A/B/C/D answer tokens.
        """
        if self.fewshot_eval_dataset is None:
            return 0.0

        device = self.device
        loader = DataLoader(
            self.fewshot_eval_dataset,
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            collate_fn=self.fewshot_eval_dataset.collate,
            num_workers=0,
            drop_last=False,
        )

        correct = 0
        total = 0

        answer_token_ids = self.fewshot_eval_dataset.answer_token_ids
        candidate_ids = torch.tensor(
            [answer_token_ids[l] for l in ["A", "B", "C", "D"]],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            for model_batch, no_model_batch in loader:
                if model_batch is None:
                    continue
                self.fewshot_eval_dataset.move_to_device(model_batch, no_model_batch, device)

                outputs = self.model(**model_batch, use_cache=False)
                logits = outputs.logits  # [B, L, V]

                # Get logits at the last non-padding position for each sample
                # With left-padding, the last token is always at position L-1
                last_logits = logits[:, -1, :]  # [B, V]

                # Extract logits for candidate answer tokens
                candidate_logits = last_logits[:, candidate_ids]  # [B, 4]
                preds = candidate_logits.argmax(dim=-1)  # [B] indices 0-3

                # Map target labels to indices
                label_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
                target_indices = torch.tensor(
                    [label_to_idx[l] for l in no_model_batch["target_label"]],
                    dtype=torch.long,
                    device=device,
                )

                correct += (preds == target_indices).sum().item()
                total += len(no_model_batch["target_label"])

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, global_step: int, final: bool = False):
        tag = "final" if final else f"step_{global_step}"
        save_path = os.path.join(self.cfg.training.save_dir, tag)

        if self.use_deepspeed:
            # DeepSpeed ZeRO-3: use engine's save method which handles
            # gathering sharded params across ranks automatically.
            # save_16bit_model saves a consolidated fp16/bf16 HF checkpoint.
            self.model.save_checkpoint(self.cfg.training.save_dir, tag=tag)
            if self.rank == 0:
                # Also save a HuggingFace-compatible model for easy loading
                hf_save_path = os.path.join(save_path, "hf_model")
                os.makedirs(hf_save_path, exist_ok=True)
                # Use DeepSpeed's utility to gather and save 16-bit weights
                success = self.model.save_16bit_model(hf_save_path)
                if success:
                    self.tokenizer.save_pretrained(hf_save_path)
                    _print_rank0(f"HF checkpoint saved: {hf_save_path}", self.rank)
                else:
                    _print_rank0(f"Warning: save_16bit_model failed for {hf_save_path}", self.rank)
        else:
            if self.rank != 0:
                return
            os.makedirs(save_path, exist_ok=True)
            # Save model
            self._raw_model.save_pretrained(save_path, safe_serialization=False)
            self.tokenizer.save_pretrained(save_path)

        # Save selection state (all ranks wait, only rank 0 writes)
        if self.rank == 0:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.grad_gamma, os.path.join(save_path, "grad_gamma.pt"))
            torch.save(self.sampler.weights, os.path.join(save_path, "cluster_weights.pt"))
            _print_rank0(f"Checkpoint saved: {save_path}", self.rank)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str, step: int):
        full_msg = f"[step={step:06d}] {msg}"
        _print_rank0(full_msg, self.rank)
        if self.rank == 0:
            _save_rank0(full_msg, self.log_file)


# ======================================================================
# Collate wrapper that injects sample indices into the batch
# ======================================================================

class IndexInjectingDataset(torch.utils.data.Dataset):
    """
    Wraps ClusterDataset to inject sample indices into each batch via collate.
    The '__indices__' key is added to model_batch and consumed by the trainer.
    """

    def __init__(self, cluster_dataset: ClusterDataset):
        self.ds = cluster_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        idx, data = self.ds[index]
        return index, idx, data  # (global_index, original_idx, token_ids)

    def collate(self, samples):
        global_indices = [s[0] for s in samples]
        inner_samples = [(s[1], s[2]) for s in samples]
        model_batch, no_model_batch = self.ds.collate(inner_samples)
        if model_batch is not None:
            model_batch["__indices__"] = torch.tensor(global_indices, dtype=torch.long)
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        return self.ds.move_to_device(model_batch, no_model_batch, device)

    @property
    def cluster_ids(self):
        return self.ds.cluster_ids
