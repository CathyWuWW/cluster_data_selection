"""
Summarize Exp-mc results across 3 configs x 3 seeds.

Inputs (under <OUT_ROOT>/eval/):
  - lm/<run>.json         : {math, code, weighted_mc, ...}
  - gsm8k/<run>.json      : {accuracy}
  - mbpp/<run>.json       : {pass_at_1}
  - humaneval/<run>.json  : {pass_at_1}

Run name format: hidden_{base|utility_meta|random_meta}_s{seed}

Outputs (printed + saved as <OUT_ROOT>/summary.{md,json}):
  1. Per-run table.
  2. Per-config (3 seed) mean + std for every metric.
  3. Δ vs base (mean and z-score) for utility_meta and random_meta.
  4. Two-sided paired t-test (utility_meta vs base) per metric.
  5. PMP cluster-weight diagnostic:
       - For each utility_meta run, top-K meta_cluster sample share
         (mean over training) vs raw utility ranking → does PMP
         actually push more mass to high-utility meta clusters?
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

REPO_ROOT = Path("/jizhicfs/wuyanning/cluster_data_selection")
PTR = REPO_ROOT / "outputs" / "latest_exp_mc_out_root.txt"

CONFIGS = ["hidden_base", "hidden_utility_meta", "hidden_random_meta"]


def load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_run_name(name: str) -> Optional[tuple[str, int]]:
    for c in CONFIGS:
        prefix = c + "_s"
        if name.startswith(prefix):
            try:
                return c, int(name[len(prefix):])
            except ValueError:
                return None
    return None


# ---------- t-test helpers (paired, no scipy required) ----------
def paired_t_pvalue_two_sided(diffs: List[float]) -> Optional[float]:
    """Approx two-sided p-value for paired t. Falls back to None if n<2."""
    n = len(diffs)
    if n < 2:
        return None
    m = mean(diffs)
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    if var <= 0:
        return 1.0 if m == 0 else 0.0
    se = math.sqrt(var / n)
    t = m / se
    # crude two-sided p-value via normal approx (n=3 we have df=2 -> Student-t,
    # but to avoid scipy we report normal-approx p; flag the small-n caveat).
    # For n=3 the 95% critical t is ~4.30 vs normal 1.96, so this normal p
    # is OPTIMISTIC (too small). We report it as a rough indicator only.
    p = math.erfc(abs(t) / math.sqrt(2))
    return p


def fmt(x: Optional[float], width: int = 8, decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'  -':>{width}}"
    return f"{x:>{width}.{decimals}f}"


def gather_eval(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return {run_name: {metric: value}}."""
    rows: Dict[str, Dict[str, Any]] = defaultdict(dict)

    # LM losses
    for p in (eval_dir / "lm").glob("*.json"):
        name = p.stem
        d = load_json(p) or {}
        rows[name]["math_loss"] = d.get("math")
        rows[name]["code_loss"] = d.get("code")
        rows[name]["mc_weighted_loss"] = d.get("weighted_mc")
    # GSM8K
    for p in (eval_dir / "gsm8k").glob("*.json"):
        d = load_json(p) or {}
        rows[p.stem]["gsm8k_acc"] = d.get("accuracy")
    # MBPP
    for p in (eval_dir / "mbpp").glob("*.json"):
        d = load_json(p) or {}
        rows[p.stem]["mbpp_pass1"] = d.get("pass_at_1")
    # HumanEval
    for p in (eval_dir / "humaneval").glob("*.json"):
        d = load_json(p) or {}
        rows[p.stem]["humaneval_pass1"] = d.get("pass_at_1")

    return rows


METRIC_ORDER = [
    "math_loss",
    "code_loss",
    "mc_weighted_loss",
    "gsm8k_acc",
    "mbpp_pass1",
    "humaneval_pass1",
]
LOWER_IS_BETTER = {"math_loss", "code_loss", "mc_weighted_loss"}


def aggregate_by_config(rows: Dict[str, Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {config: {metric: {n, mean, std}}}."""
    by_cfg: Dict[str, Dict[str, List[float]]] = {c: defaultdict(list) for c in CONFIGS}
    for name, metrics in rows.items():
        parsed = parse_run_name(name)
        if not parsed:
            continue
        cfg, _ = parsed
        for m, v in metrics.items():
            if v is not None:
                by_cfg[cfg][m].append(float(v))
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cfg, metric_lists in by_cfg.items():
        out[cfg] = {}
        for m, vals in metric_lists.items():
            out[cfg][m] = {
                "n": len(vals),
                "mean": mean(vals) if vals else float("nan"),
                "std": (pstdev(vals) if len(vals) > 1 else 0.0),
            }
    return out


def paired_diffs(rows: Dict[str, Dict], cfg_a: str, cfg_b: str, metric: str) -> List[float]:
    """Pair runs by seed: cfg_a − cfg_b."""
    a = {parse_run_name(n)[1]: v.get(metric) for n, v in rows.items()
         if parse_run_name(n) and parse_run_name(n)[0] == cfg_a and v.get(metric) is not None}
    b = {parse_run_name(n)[1]: v.get(metric) for n, v in rows.items()
         if parse_run_name(n) and parse_run_name(n)[0] == cfg_b and v.get(metric) is not None}
    seeds = sorted(set(a.keys()) & set(b.keys()))
    return [a[s] - b[s] for s in seeds]


# ---------- PMP cluster-weight diagnostic ----------
def diag_cluster_weights(out_root: Path, n_meta: int = 24, top_k: int = 8) -> List[str]:
    """For each utility_meta run, check whether high-utility meta clusters
    actually receive more sample mass over training."""
    lines: List[str] = []
    util_meta_dir = out_root / "meta_clusters" / f"utility_m{n_meta}"
    summary_csv = util_meta_dir / "meta_cluster_summary.csv"
    if not summary_csv.exists():
        lines.append(f"[diag] no meta_cluster_summary at {summary_csv}; skip.")
        return lines
    # Build mc_score = mean_raw_math + mean_raw_code per meta_cluster_id.
    import csv
    mc_score: Dict[int, float] = {}
    with summary_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = int(r["meta_cluster_id"])
            mc_score[cid] = float(r["mean_raw_math"]) + float(r["mean_raw_code"])
    ranked_by_score = sorted(mc_score.items(), key=lambda x: -x[1])
    top_ids = [cid for cid, _ in ranked_by_score[:top_k]]
    bot_ids = [cid for cid, _ in ranked_by_score[-top_k:]]
    lines.append(
        f"[diag] utility_meta has {n_meta} meta-clusters. "
        f"Top-{top_k} by (math+code) score: {top_ids}; "
        f"Bottom-{top_k}: {bot_ids}"
    )

    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        return lines
    for rd in sorted(runs_dir.iterdir()):
        if not rd.is_dir():
            continue
        parsed = parse_run_name(rd.name)
        if not parsed:
            continue
        cfg, seed = parsed
        if cfg != "hidden_utility_meta":
            continue
        hist = rd / "cluster_weight_history.jsonl"
        if not hist.exists():
            lines.append(f"  - {rd.name}: no cluster_weight_history.jsonl")
            continue
        # Average sampling weight across all logged steps for top vs bottom.
        top_sums, bot_sums, n_steps = 0.0, 0.0, 0
        with hist.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                w = rec.get("weights") or rec.get("cluster_weights")
                if not w:
                    continue
                # weights is dict[str(cluster_id) -> prob] or list-like
                if isinstance(w, dict):
                    get = lambda cid: float(w.get(str(cid), w.get(cid, 0.0)))
                elif isinstance(w, list):
                    get = lambda cid: float(w[cid]) if cid < len(w) else 0.0
                else:
                    continue
                top_sums += sum(get(cid) for cid in top_ids)
                bot_sums += sum(get(cid) for cid in bot_ids)
                n_steps += 1
        if n_steps == 0:
            lines.append(f"  - {rd.name}: history empty")
            continue
        top_mean = top_sums / n_steps
        bot_mean = bot_sums / n_steps
        uniform_share = top_k / n_meta
        lines.append(
            f"  - {rd.name}: avg_top{top_k}_share={top_mean:.4f} "
            f"avg_bot{top_k}_share={bot_mean:.4f} "
            f"(uniform={uniform_share:.4f})  "
            f"top-bot={top_mean - bot_mean:+.4f}"
        )
    return lines


def main() -> None:
    if not PTR.exists():
        print(f"[err] missing {PTR}")
        sys.exit(1)
    out_root = Path(PTR.read_text().strip())
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    eval_dir = out_root / "eval"
    print(f"[summary] OUT_ROOT={out_root}")
    print(f"[summary] EVAL_DIR={eval_dir}")
    if not eval_dir.exists():
        print(f"[err] no eval dir yet: {eval_dir}")
        sys.exit(1)

    rows = gather_eval(eval_dir)
    if not rows:
        print("[err] no eval results found")
        sys.exit(1)

    md: List[str] = []
    md.append(f"# Exp-mc summary  ({out_root.name})\n")

    # 1. Per-run table
    md.append("## 1. Per-run metrics\n")
    header = ["run"] + METRIC_ORDER
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for name in sorted(rows.keys()):
        cells = [name]
        for m in METRIC_ORDER:
            v = rows[name].get(m)
            cells.append(fmt(v, 0, 4).strip() if v is not None else "-")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # 2. Per-config mean ± std
    by_cfg = aggregate_by_config(rows)
    md.append("## 2. Per-config (mean ± std over seeds)\n")
    md.append("| config | " + " | ".join(METRIC_ORDER) + " |")
    md.append("| --- | " + " | ".join(["---"] * len(METRIC_ORDER)) + " |")
    for cfg in CONFIGS:
        cells = [cfg]
        for m in METRIC_ORDER:
            stat = by_cfg.get(cfg, {}).get(m)
            if stat and not math.isnan(stat["mean"]):
                cells.append(f"{stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['n']})")
            else:
                cells.append("-")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # 3. Δ vs base (mean and per-seed-std-normalized z)
    md.append("## 3. Δ vs hidden_base  (utility_meta − base, random_meta − base)\n")
    md.append(
        "Sign convention: for losses, Δ<0 means improvement; "
        "for accuracies, Δ>0 means improvement.  "
        "z = mean(Δ) / std(base across seeds)  (rough effect size).\n"
    )
    md.append("| config_vs_base | metric | mean Δ | std Δ | base std | z (mean Δ / base std) | paired-t p≈ |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for cfg in ["hidden_utility_meta", "hidden_random_meta"]:
        for m in METRIC_ORDER:
            diffs = paired_diffs(rows, cfg, "hidden_base", m)
            if not diffs:
                continue
            d_mean = mean(diffs)
            d_std = pstdev(diffs) if len(diffs) > 1 else 0.0
            base_std = by_cfg.get("hidden_base", {}).get(m, {}).get("std", 0.0)
            z = (d_mean / base_std) if base_std > 0 else float("nan")
            p = paired_t_pvalue_two_sided(diffs)
            md.append(
                f"| {cfg} vs base | {m} | "
                f"{d_mean:+.4f} | {d_std:.4f} | {base_std:.4f} | "
                f"{z:+.2f} | {('-' if p is None else f'{p:.3f}')} |"
            )
    md.append("")
    md.append(
        "_Note_: with only n=3 seeds, the normal-approx p-value above is "
        "OPTIMISTIC (true Student-t critical value is ~2× higher). Treat "
        "p<0.05 here as 'directionally interesting', not as conclusive.\n"
    )

    # 4. Cluster-weight diagnostic
    md.append("## 4. PMP cluster-weight diagnostic (utility_meta only)\n")
    md.append("```")
    md.extend(diag_cluster_weights(out_root, n_meta=24, top_k=8))
    md.append("```\n")
    md.append(
        "If `top-bot` is clearly positive (e.g. > 0.05), PMP is actually "
        "shifting sampling mass toward high-utility meta-clusters — the "
        "intended mechanism is firing.\n"
    )

    out_md = out_root / "summary.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[summary] wrote {out_md}")

    # raw JSON dump
    out_json = out_root / "summary.json"
    out_json.write_text(json.dumps({
        "rows": rows, "by_cfg": by_cfg,
    }, indent=2, default=lambda o: None), encoding="utf-8")
    print(f"[summary] wrote {out_json}")


if __name__ == "__main__":
    main()
