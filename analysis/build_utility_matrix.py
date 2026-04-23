"""
Build Experiment-2 utility matrices from per-ability training runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class RunStats:
    ability: str
    run_dir: Path
    n_clusters: int
    n_updates: int
    final_step: int
    sum_delta: np.ndarray
    final_weight: np.ndarray
    avg_tail_weight: np.ndarray
    positive_fraction: np.ndarray
    final_grad_gamma: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cluster_weight_history.jsonl runs into utility matrices."
    )
    parser.add_argument("--runs-root", required=True, help="Root directory containing one subdir per ability run.")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV/JSON summaries.")
    parser.add_argument(
        "--abilities",
        default="",
        help="Optional comma-separated ability order. Defaults to discovered abilities.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Fraction of final PMP updates used for avg_tail_weight.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def discover_run_dirs(runs_root: Path) -> List[Path]:
    return sorted(
        path for path in runs_root.iterdir()
        if path.is_dir() and (path / "cluster_weight_history.jsonl").is_file()
    )


def load_ability(run_dir: Path) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        ability = str(metadata.get("ability", "")).strip()
        if ability:
            return ability
    return run_dir.name


def load_history(run_dir: Path) -> List[Dict]:
    path = run_dir / "cluster_weight_history.jsonl"
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "pmp_update":
                records.append(record)
    if not records:
        raise ValueError(f"No pmp_update records found in {path}")
    return records


def stats_from_run(run_dir: Path, tail_fraction: float) -> RunStats:
    ability = load_ability(run_dir)
    records = load_history(run_dir)
    n_clusters = int(records[-1]["n_clusters"])

    deltas = []
    weights = []
    for record in records:
        delta = record.get("grad_gamma_delta")
        if delta is None:
            raise ValueError(f"Missing grad_gamma_delta in {run_dir}")
        deltas.append(np.asarray(delta, dtype=np.float32))
        weights.append(np.asarray(record["weights"], dtype=np.float32))

    for arr in deltas + weights:
        if arr.shape != (n_clusters,):
            raise ValueError(
                f"Inconsistent cluster count in {run_dir}: expected {n_clusters}, got {arr.shape}"
            )

    deltas_np = np.stack(deltas, axis=0)
    weights_np = np.stack(weights, axis=0)
    tail_count = max(1, int(np.ceil(len(records) * tail_fraction)))

    return RunStats(
        ability=ability,
        run_dir=run_dir,
        n_clusters=n_clusters,
        n_updates=len(records),
        final_step=int(records[-1]["step"]),
        sum_delta=deltas_np.sum(axis=0),
        final_weight=weights_np[-1],
        avg_tail_weight=weights_np[-tail_count:].mean(axis=0),
        positive_fraction=(deltas_np > 0).mean(axis=0),
        final_grad_gamma=np.asarray(records[-1]["grad_gamma"], dtype=np.float32),
    )


def aggregate_runs(stats_list: Sequence[RunStats], ability_order: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, List[RunStats]] = {ability: [] for ability in ability_order}
    for stats in stats_list:
        grouped.setdefault(stats.ability, []).append(stats)

    n_clusters = stats_list[0].n_clusters
    metrics = {
        "sum_delta": {},
        "final_weight": {},
        "avg_tail_weight": {},
        "positive_fraction": {},
        "final_grad_gamma": {},
    }

    for ability in ability_order:
        runs = grouped.get(ability, [])
        if not runs:
            raise ValueError(f"No runs found for ability '{ability}'")
        for run in runs:
            if run.n_clusters != n_clusters:
                raise ValueError(
                    f"Cluster count mismatch for ability '{ability}': "
                    f"{run.n_clusters} vs {n_clusters}"
                )

        metrics["sum_delta"][ability] = np.mean([run.sum_delta for run in runs], axis=0)
        metrics["final_weight"][ability] = np.mean([run.final_weight for run in runs], axis=0)
        metrics["avg_tail_weight"][ability] = np.mean([run.avg_tail_weight for run in runs], axis=0)
        metrics["positive_fraction"][ability] = np.mean([run.positive_fraction for run in runs], axis=0)
        metrics["final_grad_gamma"][ability] = np.mean([run.final_grad_gamma for run in runs], axis=0)

    return metrics


def write_matrix_csv(path: Path, ability_order: Sequence[str], metric_name: str, matrix: Dict[str, np.ndarray]) -> None:
    n_clusters = len(next(iter(matrix.values())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", *ability_order])
        for cluster_id in range(n_clusters):
            writer.writerow(
                [cluster_id, *[float(matrix[ability][cluster_id]) for ability in ability_order]]
            )


def write_long_csv(path: Path, ability_order: Sequence[str], metrics: Dict[str, Dict[str, np.ndarray]]) -> None:
    n_clusters = len(next(iter(metrics["sum_delta"].values())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ability",
                "cluster_id",
                "sum_delta",
                "final_weight",
                "avg_tail_weight",
                "positive_fraction",
                "final_grad_gamma",
            ]
        )
        for ability in ability_order:
            for cluster_id in range(n_clusters):
                writer.writerow(
                    [
                        ability,
                        cluster_id,
                        float(metrics["sum_delta"][ability][cluster_id]),
                        float(metrics["final_weight"][ability][cluster_id]),
                        float(metrics["avg_tail_weight"][ability][cluster_id]),
                        float(metrics["positive_fraction"][ability][cluster_id]),
                        float(metrics["final_grad_gamma"][ability][cluster_id]),
                    ]
                )


def write_cluster_summary(path: Path, ability_order: Sequence[str], metrics: Dict[str, Dict[str, np.ndarray]]) -> None:
    abilities = list(ability_order)
    n_clusters = len(next(iter(metrics["sum_delta"].values())))
    score_matrix = np.stack([metrics["sum_delta"][ability] for ability in abilities], axis=1)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cluster_id",
                "dominant_ability",
                "dominant_sum_delta",
                "runner_up_ability",
                "runner_up_sum_delta",
                "dominance_margin",
                "utility_l2_norm",
            ]
        )
        for cluster_id in range(n_clusters):
            row = score_matrix[cluster_id]
            order = np.argsort(row)[::-1]
            best_idx = int(order[0])
            second_idx = int(order[1]) if len(order) > 1 else int(order[0])
            best_val = float(row[best_idx])
            second_val = float(row[second_idx])
            writer.writerow(
                [
                    cluster_id,
                    abilities[best_idx],
                    best_val,
                    abilities[second_idx],
                    second_val,
                    best_val - second_val,
                    float(np.linalg.norm(row)),
                ]
            )


def write_run_summary(path: Path, stats_list: Sequence[RunStats]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ability", "run_dir", "n_clusters", "n_updates", "final_step"])
        for stats in stats_list:
            writer.writerow(
                [
                    stats.ability,
                    str(stats.run_dir),
                    stats.n_clusters,
                    stats.n_updates,
                    stats.final_step,
                ]
            )


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with cluster_weight_history.jsonl under {runs_root}")

    stats_list = [stats_from_run(run_dir, tail_fraction=args.tail_fraction) for run_dir in run_dirs]
    ability_order = parse_csv_list(args.abilities) or sorted({stats.ability for stats in stats_list})
    metrics = aggregate_runs(stats_list, ability_order)

    write_run_summary(out_dir / "run_summary.csv", stats_list)
    write_long_csv(out_dir / "utility_long.csv", ability_order, metrics)
    write_cluster_summary(out_dir / "cluster_summary.csv", ability_order, metrics)

    for metric_name, matrix in metrics.items():
        write_matrix_csv(out_dir / f"utility_matrix_{metric_name}.csv", ability_order, metric_name, matrix)

    np.savez(
        out_dir / "utility_matrices.npz",
        abilities=np.asarray(ability_order, dtype=object),
        **{
            f"{metric_name}__{ability}": matrix[ability]
            for metric_name, matrix in metrics.items()
            for ability in ability_order
        },
    )

    metadata = {
        "runs_root": str(runs_root),
        "abilities": ability_order,
        "tail_fraction": args.tail_fraction,
        "n_runs": len(stats_list),
        "n_clusters": stats_list[0].n_clusters,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote utility matrices to {out_dir}")


if __name__ == "__main__":
    main()
