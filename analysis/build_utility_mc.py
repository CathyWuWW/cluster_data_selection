"""
Build a math+code-only utility matrix from the original 3-domain utility CSV
(`utility_matrix_sum_delta.csv`).

Outputs three CSVs:
  - utility_matrix_mc_sum_delta.csv : columns [cluster_id, math, code]
  - utility_matrix_mc_mean.csv      : columns [cluster_id, mc] = (math+code)/2
  - cluster_ranking_mc.csv          : sorted by mc descending, top-K view

The first one is what we feed to `analysis/build_meta_clusters.py` so that
KMeans operates on a 2-D (math, code) feature space (after z-score) instead
of the original 3-D one with reading.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-csv",
        default="outputs/utility_vector_hidden_l2rm1pc_real_20260427_194520/utility_matrix/utility_matrix_sum_delta.csv",
    )
    p.add_argument("--out-dir", default="outputs/utility_mc_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(in_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"empty: {in_path}")

    # 2-D matrix (math, code) — keep raw values, downstream will z-score.
    out_2d = out_dir / "utility_matrix_mc_sum_delta.csv"
    with out_2d.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "math", "code"])
        for r in rows:
            w.writerow([r["cluster_id"], r["math"], r["code"]])

    # 1-D scalar utility (math+code)/2 — for sanity ranking only.
    scored = []
    for r in rows:
        m = float(r["math"])
        c = float(r["code"])
        scored.append((int(r["cluster_id"]), m, c, (m + c) / 2.0))

    out_1d = out_dir / "utility_matrix_mc_mean.csv"
    with out_1d.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "mc"])
        for cid, _m, _c, mc in scored:
            w.writerow([cid, f"{mc:.6f}"])

    ranked = sorted(scored, key=lambda x: x[3], reverse=True)
    out_rank = out_dir / "cluster_ranking_mc.csv"
    with out_rank.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cluster_id", "math", "code", "mc"])
        for i, (cid, m, c, mc) in enumerate(ranked):
            w.writerow([i, cid, f"{m:.4f}", f"{c:.4f}", f"{mc:.4f}"])

    # Quick stdout summary
    n = len(scored)
    pos = sum(1 for x in scored if x[3] > 0)
    print(f"[done] wrote 3 files under {out_dir}")
    print(f"  total clusters: {n}")
    print(f"  positive mc   : {pos} ({pos/n:.1%})")
    print("  top 10 by mc:")
    for i in range(min(10, n)):
        cid, m, c, mc = ranked[i]
        print(f"    rank {i:>2}  cid={cid:>3}  math={m:+.2f}  code={c:+.2f}  mc={mc:+.2f}")
    print("  bottom 5 by mc:")
    for i in range(max(0, n - 5), n):
        cid, m, c, mc = ranked[i]
        print(f"    rank {i:>2}  cid={cid:>3}  math={m:+.2f}  code={c:+.2f}  mc={mc:+.2f}")


if __name__ == "__main__":
    main()
