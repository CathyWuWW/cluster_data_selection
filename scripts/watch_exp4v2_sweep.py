from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path('/jizhicfs/wuyanning/cluster_data_selection')
OUTPUTS = ROOT / 'outputs'
RUN_ROOTS = {
    'alpha0':   OUTPUTS / 'exp4v2_hidden_l2rm1pc_real_prior_alpha0_20260429_2240_s42',
    'alpha0p2': OUTPUTS / 'exp4v2_hidden_l2rm1pc_real_prior_alpha0p2_20260429_2240_s43',
    'alpha0p5': OUTPUTS / 'exp4v2_hidden_l2rm1pc_real_prior_alpha0p5_20260429_2240_s44',
    'alpha1p0': OUTPUTS / 'exp4v2_hidden_l2rm1pc_real_prior_alpha1p0_20260429_2240_s45',
}
RUN_NAME_BY_TAG = {
    'alpha0':   'prior_alpha0',
    'alpha0p2': 'prior_alpha0p2',
    'alpha0p5': 'prior_alpha0p5',
    'alpha1p0': 'prior_alpha1p0',
}
ALPHA_BY_TAG = {
    'alpha0':   0.0,
    'alpha0p2': 0.2,
    'alpha0p5': 0.5,
    'alpha1p0': 1.0,
}

BASE_WEIGHTED = 1.8435
BEST_EXP3 = 1.8367
POLL_SECONDS = 300
STATUS_JSON = OUTPUTS / 'exp4v2_sweep_watch_status.json'
SUMMARY_MD = OUTPUTS / 'exp4v2_sweep_final_summary.md'
SUMMARY_JSON = OUTPUTS / 'exp4v2_sweep_final_summary.json'
LOG_PREFIX = '[exp4v2-watch]'

STEP_RE = re.compile(r'\[step=(\d{6})\] step=(\d+)/300')
EVAL_RE = re.compile(
    r'\[step=(\d{6})\] eval: math=(\S+)\s+math_ppl=(\S+)\s+reading=(\S+)\s+'
    r'reading_ppl=(\S+)\s+code=(\S+)\s+code_ppl=(\S+)\s+weighted=(\S+)\s+weighted_ppl=(\S+)'
)
DONE_RE = re.compile(r'\[step=000300\] Training complete\.')
PMP_RE = re.compile(
    r'\[PMP\] weights: min=(\S+), max=(\S+), entropy=(\S+), alive=(\d+)/(\d+), dropped=(\d+)'
)
PRIOR_RE = re.compile(
    r'\[utility_prior\] enabled: .*alpha=(\S+) sign=(\S+) agg=(\S+) norm=(\S+) '
    r'grad_gamma_init: min=(\S+) max=(\S+) norm=(\S+) ; '
    r'sampler\.weights@init: min=(\S+) max=(\S+) std=(\S+)'
)


@dataclass
class RunStatus:
    status: str
    current_step: Optional[int] = None
    last_eval_step: Optional[int] = None
    weighted: Optional[float] = None
    math: Optional[float] = None
    reading: Optional[float] = None
    code: Optional[float] = None
    entropy: Optional[float] = None
    max_weight: Optional[float] = None
    alive: Optional[str] = None
    prior_alpha: Optional[float] = None
    prior_init_norm: Optional[float] = None
    sampler_init_max: Optional[float] = None
    sampler_init_std: Optional[float] = None


def parse_run(log_path: Path) -> RunStatus:
    if not log_path.exists():
        return RunStatus(status='missing')
    text = log_path.read_text(errors='ignore')
    steps = STEP_RE.findall(text)
    evals = EVAL_RE.findall(text)
    pmps = PMP_RE.findall(text)
    prior = PRIOR_RE.search(text)
    done = bool(DONE_RE.search(text))
    status = 'done' if done else 'running'
    current_step = int(steps[-1][1]) if steps else None
    if evals:
        last = evals[-1]
        last_eval_step = int(last[0])
        math_v = float(last[1])
        reading_v = float(last[3])
        code_v = float(last[5])
        weighted = float(last[7])
    else:
        last_eval_step = None
        math_v = reading_v = code_v = weighted = None
    if pmps:
        p = pmps[-1]
        max_weight = float(p[1])
        entropy = float(p[2])
        alive = f'{p[3]}/{p[4]}'
    else:
        max_weight = entropy = None
        alive = None
    return RunStatus(
        status=status,
        current_step=current_step,
        last_eval_step=last_eval_step,
        weighted=weighted,
        math=math_v,
        reading=reading_v,
        code=code_v,
        entropy=entropy,
        max_weight=max_weight,
        alive=alive,
        prior_alpha=float(prior.group(1)) if prior else None,
        prior_init_norm=float(prior.group(7)) if prior else None,
        sampler_init_max=float(prior.group(9)) if prior else None,
        sampler_init_std=float(prior.group(10)) if prior else None,
    )


def snapshot() -> dict:
    snap = {}
    for tag, root in RUN_ROOTS.items():
        log = root / 'runs' / RUN_NAME_BY_TAG[tag] / 'train.log'
        snap[tag] = asdict(parse_run(log))
        snap[tag]['expected_alpha'] = ALPHA_BY_TAG[tag]
    return snap


def all_done(snap: dict) -> bool:
    return all(snap[tag]['status'] == 'done' for tag in RUN_ROOTS)


def _fmt(x, spec='.4f'):
    if x is None:
        return '-'
    return format(x, spec)


def build_summary(snap: dict) -> tuple[str, dict]:
    rows = []
    for tag in ('alpha0', 'alpha0p2', 'alpha0p5', 'alpha1p0'):
        s = snap[tag]
        rows.append({
            'tag': tag,
            'alpha': ALPHA_BY_TAG[tag],
            'weighted': s['weighted'],
            'math': s['math'],
            'reading': s['reading'],
            'code': s['code'],
            'entropy': s['entropy'],
            'max_weight': s['max_weight'],
            'alive': s['alive'],
            'prior_init_norm': s['prior_init_norm'],
            'sampler_init_max': s['sampler_init_max'],
            'sampler_init_std': s['sampler_init_std'],
            'delta_vs_base': round(s['weighted'] - BASE_WEIGHTED, 4) if s['weighted'] is not None else None,
            'delta_vs_best_exp3': round(s['weighted'] - BEST_EXP3, 4) if s['weighted'] is not None else None,
        })
    valid_rows = [r for r in rows if r['weighted'] is not None]
    best = min(valid_rows, key=lambda r: r['weighted']) if valid_rows else None
    js = {
        'base_weighted': BASE_WEIGHTED,
        'best_exp3_weighted': BEST_EXP3,
        'best_alpha': best['alpha'] if best else None,
        'best_weighted': best['weighted'] if best else None,
        'rows': rows,
    }
    lines = [
        '### Exp4-v2 utility-prior sweep 自动汇总',
        '',
        f'- **base weighted (87-way, no prior)**: {BASE_WEIGHTED:.4f}',
        f'- **best Exp3 weighted (utility_meta@m=24)**: {BEST_EXP3:.4f}',
        f'- **best alpha**: {best["alpha"] if best else "N/A"}',
        f'- **best weighted**: {_fmt(best["weighted"]) if best else "N/A"}',
        '',
        '| alpha | weighted | math | reading | code | Δ vs base | Δ vs Exp3 best | entropy | max_w | sampler_init_max | sampler_init_std | init_norm |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['alpha']} | {_fmt(r['weighted'])} | {_fmt(r['math'])} | {_fmt(r['reading'])} | {_fmt(r['code'])} | "
            f"{_fmt(r['delta_vs_base'], '+.4f')} | {_fmt(r['delta_vs_best_exp3'], '+.4f')} | "
            f"{_fmt(r['entropy'], '.3f')} | {_fmt(r['max_weight'])} | "
            f"{_fmt(r['sampler_init_max'])} | {_fmt(r['sampler_init_std'])} | {_fmt(r['prior_init_norm'], '.3f')} |"
        )
    lines += [
        '',
        '### 自动结论提示',
        '',
        '- **结论 1**：随 `alpha` 变化，`weighted` 是单调还是非单调？',
        '- **结论 2**：是否存在某个 `alpha` 同时优于 `base` 和 `Exp3 best`？',
        '- **结论 3**：观察 `entropy` 与 `max_weight`，prior 是否带来过度倾斜。',
        '- **结论 4**：与上一轮 buggy 实现对比，`sampler_init_max/std` 现在应随 `alpha` 显著变化。',
        '',
    ]
    return '\n'.join(lines), js


def main() -> None:
    print(f'{LOG_PREFIX} start watching {len(RUN_ROOTS)} prior runs', flush=True)
    while True:
        snap = snapshot()
        STATUS_JSON.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding='utf-8')
        progress = []
        for tag in ('alpha0', 'alpha0p2', 'alpha0p5', 'alpha1p0'):
            s = snap[tag]
            progress.append(f"{tag}: status={s['status']} step={s['current_step']}")
        print(f"{LOG_PREFIX} " + ' | '.join(progress), flush=True)
        if all_done(snap):
            try:
                md, js = build_summary(snap)
                SUMMARY_MD.write_text(md, encoding='utf-8')
                SUMMARY_JSON.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'{LOG_PREFIX} all runs finished; wrote {SUMMARY_MD.name} and {SUMMARY_JSON.name}', flush=True)
            except Exception as e:
                print(f'{LOG_PREFIX} summary failed: {e}', flush=True)
            break
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
