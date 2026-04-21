"""
Plot representation comparison summaries.

Reads outputs from representation_comparison.py and writes a few compact PNG
figures for quick experiment reports.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot representation comparison CSVs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def short_label(name: str, max_len: int = 48) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def barh(rows: List[Dict[str, str]], metric: str, title: str, path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: as_float(r, metric), reverse=True)[:top_n]
    labels = [short_label(row["name"]) for row in rows][::-1]
    values = [as_float(row, metric) for row in rows][::-1]
    colors = [family_color(row.get("family", "")) for row in rows][::-1]

    height = max(4.0, 0.34 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def family_color(family: str) -> str:
    return {
        "hidden": "#4C78A8",
        "sentence": "#54A24B",
        "lexical": "#F58518",
        "random": "#B279A2",
        "length": "#E45756",
        "source": "#72B7B2",
    }.get(family, "#9D9D9D")


def scatter_pc_vs_source(rows: List[Dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        ax.scatter(
            as_float(row, "pc1_variance_ratio"),
            as_float(row, "same_source_neighbor_rate"),
            color=family_color(row.get("family", "")),
            alpha=0.85,
            s=42,
        )
    ax.set_xlabel("pc1_variance_ratio")
    ax.set_ylabel("same_source_neighbor_rate")
    ax.set_title("Anisotropy vs. Source-Neighborhood Rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_k_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for row in rows:
        rep = row["representation"]
        score = as_float(row, "silhouette_cosine", default=-999.0)
        if rep not in best or score > as_float(best[rep], "silhouette_cosine", default=-999.0):
            best[rep] = row
    return list(best.values())


def barh_cluster(rows: List[Dict[str, str]], metric: str, title: str, path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: as_float(r, metric), reverse=True)[:top_n]
    labels = [short_label(f"{row['representation']} (k={row['k']})") for row in rows][::-1]
    values = [as_float(row, metric) for row in rows][::-1]
    colors = [family_color(row.get("family", "")) for row in rows][::-1]

    height = max(4.0, 0.34 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_rows = read_csv(run_dir / "representation_summary.csv")
    cluster_rows = read_csv(run_dir / "clustering_summary.csv")
    best_clusters = best_k_rows(cluster_rows)

    barh(
        rep_rows,
        metric="same_source_neighbor_rate",
        title="Same-Source Rate of Top-K Neighbors",
        path=out_dir / "same_source_neighbor_rate.png",
        top_n=args.top_n,
    )
    barh(
        rep_rows,
        metric="same_source_lift_vs_random",
        title="Same-Source Lift vs. Random Baseline",
        path=out_dir / "same_source_lift_vs_random.png",
        top_n=args.top_n,
    )
    barh(
        rep_rows,
        metric="pc1_variance_ratio",
        title="PC1 Variance Ratio",
        path=out_dir / "pc1_variance_ratio.png",
        top_n=args.top_n,
    )
    scatter_pc_vs_source(rep_rows, out_dir / "pc1_vs_same_source.png")
    barh_cluster(
        best_clusters,
        metric="silhouette_cosine",
        title="Best KMeans Silhouette by Representation",
        path=out_dir / "best_silhouette_cosine.png",
        top_n=args.top_n,
    )
    barh_cluster(
        best_clusters,
        metric="weighted_source_purity",
        title="Best KMeans Source Purity by Representation",
        path=out_dir / "best_weighted_source_purity.png",
        top_n=args.top_n,
    )
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
