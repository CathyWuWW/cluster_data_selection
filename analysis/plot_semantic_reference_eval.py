"""
Plot semantic-reference evaluation results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot semantic reference metrics.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def short_name(name: str) -> str:
    return (
        name.replace("hidden_", "")
        .replace("layer", "L")
        .replace("_mean_l2", " mean")
        .replace("_rm", " rm")
        .replace("ref_intfloat-e5-small-v2", "E5")
        .replace("ref_baai-bge-small-en-v1.5", "BGE")
        .replace("random_normal_d768", "random")
        .replace("tfidf_word_5000", "TF-IDF")
    )


def family_color(family: str) -> str:
    return {
        "hidden": "#4C78A8",
        "lexical": "#F58518",
        "random": "#B279A2",
    }.get(family, "#9D9D9D")


def grouped_bar(
    rows: List[Dict[str, str]],
    metric: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    candidates = []
    for row in rows:
        if row["candidate"] not in candidates:
            candidates.append(row["candidate"])
    references = []
    for row in rows:
        if row["reference"] not in references:
            references.append(row["reference"])

    x = np.arange(len(candidates))
    width = 0.8 / max(len(references), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(candidates)), 5.4))
    for i, ref in enumerate(references):
        vals = []
        colors = []
        for candidate in candidates:
            match = next((r for r in rows if r["candidate"] == candidate and r["reference"] == ref), None)
            vals.append(as_float(match, metric) if match else 0.0)
            colors.append(family_color(match.get("candidate_family", "")) if match else "#9D9D9D")
        positions = x + (i - (len(references) - 1) / 2) * width
        bars = ax.bar(
            positions,
            vals,
            width=width,
            label=short_name(ref),
            color=colors,
            alpha=0.95 if i == 0 else 0.65,
            hatch="" if i == 0 else "//",
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([short_name(c) for c in candidates], rotation=28, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(run_dir / "semantic_reference_metrics.csv")

    all_rows = [r for r in rows if r["mode"] == "all"]
    cross_rows = [r for r in rows if r["mode"] == "cross_source"]
    overlap_metric = f"overlap_at_{args.top_k}"
    jaccard_metric = f"jaccard_at_{args.top_k}"

    grouped_bar(
        all_rows,
        metric=overlap_metric,
        title=f"Top-{args.top_k} Neighbor Overlap with Semantic References",
        path=out_dir / f"all_overlap_at_{args.top_k}.png",
    )
    grouped_bar(
        cross_rows,
        metric=overlap_metric,
        title=f"Cross-Source Top-{args.top_k} Neighbor Overlap",
        path=out_dir / f"cross_source_overlap_at_{args.top_k}.png",
    )
    grouped_bar(
        cross_rows,
        metric="spearman_mean",
        title="Cross-Source Similarity Rank Correlation",
        path=out_dir / "cross_source_spearman.png",
    )
    grouped_bar(
        cross_rows,
        metric=jaccard_metric,
        title=f"Cross-Source Top-{args.top_k} Jaccard with Semantic References",
        path=out_dir / f"cross_source_jaccard_at_{args.top_k}.png",
    )
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
