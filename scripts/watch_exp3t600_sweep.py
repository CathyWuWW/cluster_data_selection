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
    'm24': OUTPUTS / 'exp3t600_hidden_l2rm1pc_real_m24_20260430_0005_s42',
    'm32': OUTPUTS / 'exp3t600_hidden_l2rm1pc_real_m32_20260430_0005_s42',
    'm48': OUTPUTS / 'exp3t600_hidden_l2rm1pc_real_m48_20260430_0005_s42',
    'm64': OUTPUTS / 'exp3t600_hidden_l2rm1pc_real_m64_20260430_0005_s42',
}
RUN_NAMES = ('hidden_utility_meta', 'hidden_random_meta')

# Reference baselines (all on 300-step training; new 600-step results may shift).
BASE_300_WEIGHTED = 1.8435
BEST_EXP3_300_WEIGHTED = 1.8367

POLL_SECONDS = 300
TOTAL_ITERS = 600
DONE_PATTERN = re.compile(rf'\[step=00{TOTAL_ITERS:04d}\] Training complete\.')

STEP_RE = re.compile(r'\[step=(\d{6})\] step=(\d+)/' + str(TOTAL_ITERS))
EVAL_RE = re.compile(
    r'\[step=(\d{6})\] (?:\[init\] )?eval: math=(\S+)\s+math_ppl=(\S+)\s+reading=(\S+)\s+'
    r'reading_ppl=(\S+)\s+code=(\S+)\s+code_ppl=(\S+)\s+weighted=(\S+)\s+weighted_ppl=(\S+)'
)
PMP_RE = re.compile(
    r'\[PMP\] weights: min=(\S+), max=(\S+), entropy=(\S+), alive=(\d+)/(\d+), dropped=(\d+)'
)

STATUS_JSON = OUTPUTS / 'exp3t600_sweep_watch_status.json'
SUMMARY_MD = OUTPUTS / 'exp3t600_sweep_final_summary.md'
SUMMARY_JSON = OUTPUTS / 'exp3t600_sweep_final_summary.json'
LOG_PREFIX = '[exp3t600-watch]'


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


def parse_run(log_path: Path) -> RunStatus:
    if not log_path.exists():
        return RunStatus(status='missing')
    text = log_path.read_text(errors='ignore')
    steps = STEP_RE.findall(text)
    evals = EVAL_RE.findall(text)
    pmps = PMP_RE.findall(text)
    done = bool(DONE_PATTERN.search(text))
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
    )


def snapshot() -> dict:
    snap = {}
    for tag, root in RUN_ROOTS.items():
        snap[tag] = {}
        for run_name in RUN_NAMES:
            log = root / 'runs' / run_name / 'train.log'
            snap[tag][run_name] = asdict(parse_run(log))
    return snap


def all_done(snap: dict) -> bool:
    for tag in RUN_ROOTS:
        for run_name in RUN_NAMES:
            if snap[tag][run_name]['status'] != 'done':
                return False
    return True


def _fmt(x, spec='.4f'):
    return '-' if x is None else format(x, spec)


def build_summary(snap: dict) -> tuple[str, dict]:
    rows = []
    for tag in ('m24', 'm32', 'm48', 'm64'):
        util = snap[tag]['hidden_utility_meta']
        rand = snap[tag]['hidden_random_meta']
        u_w = util['weighted']
        r_w = rand['weighted']
        rows.append({
            'm': int(tag[1:]),
            'utility_weighted': u_w,
            'random_weighted': r_w,
            'utility_minus_random': round(u_w - r_w, 4) if (u_w is not None and r_w is not None) else None,
            'utility_minus_base300': round(u_w - BASE_300_WEIGHTED, 4) if u_w is not None else None,
            'utility_math': util['math'],
            'utility_reading': util['reading'],
            'utility_code': util['code'],
            'utility_entropy': util['entropy'],
            'utility_max_weight': util['max_weight'],
            'random_math': rand['math'],
            'random_reading': rand['reading'],
            'random_code': rand['code'],
            'random_entropy': rand['entropy'],
            'random_max_weight': rand['max_weight'],
        })
    valid_u = [r for r in rows if r['utility_weighted'] is not None]
    best_u = min(valid_u, key=lambda r: r['utility_weighted']) if valid_u else None
    js = {
        'total_iters': TOTAL_ITERS,
        'reference_base_300': BASE_300_WEIGHTED,
        'reference_best_exp3_300': BEST_EXP3_300_WEIGHTED,
        'best_utility_m': best_u['m'] if best_u else None,
        'best_utility_weighted': best_u['utility_weighted'] if best_u else None,
        'rows': rows,
    }
    lines = [
        f'### Exp3 (TOTAL_ITERS={TOTAL_ITERS}) m-sweep 自动汇总',
        '',
        f'- **参考: base@300步**: {BASE_300_WEIGHTED:.4f}',
        f'- **参考: Exp3 best@300步**: {BEST_EXP3_300_WEIGHTED:.4f}',
        f'- **当前最优 utility m**: {best_u["m"] if best_u else "N/A"}',
        f'- **当前最优 utility weighted**: {_fmt(best_u["utility_weighted"]) if best_u else "N/A"}',
        '',
        '| m | utility weighted | random weighted | utility-random | utility-base300 | utility(math/reading/code) | random(math/reading/code) |',
        '|---:|---:|---:|---:|---:|---|---|',
    ]
    for r in rows:
        u_str = (f"{_fmt(r['utility_math'])} / {_fmt(r['utility_reading'])} / {_fmt(r['utility_code'])}")
        rd_str = (f"{_fmt(r['random_math'])} / {_fmt(r['random_reading'])} / {_fmt(r['random_code'])}")
        lines.append(
            f"| {r['m']} | {_fmt(r['utility_weighted'])} | {_fmt(r['random_weighted'])} | "
            f"{_fmt(r['utility_minus_random'], '+.4f')} | {_fmt(r['utility_minus_base300'], '+.4f')} | "
            f"{u_str} | {rd_str} |"
        )
    lines += [
        '',
        '### 自动结论提示',
        '',
        '- **结论 1**：相比 300 步，600 步是否让 utility/random 拉开更大差距？',
        '- **结论 2**：utility 是否在更多 m 上稳定优于 random？',
        '- **结论 3**：utility 是否在某个 m 上明显优于 base@300（注意 base 还没在 600 步上重新跑）。',
        '',
    ]
    return '\n'.join(lines), js


def main() -> None:
    print(f'{LOG_PREFIX} start watching {len(RUN_ROOTS)} sweep dirs (TOTAL_ITERS={TOTAL_ITERS})', flush=True)
    while True:
        snap = snapshot()
        STATUS_JSON.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding='utf-8')
        progress = []
        for tag in ('m32', 'm48', 'm24', 'm64'):
            util = snap[tag]['hidden_utility_meta']
            rand = snap[tag]['hidden_random_meta']
            progress.append(f"{tag}: utility={util['status']}@{util['current_step']} random={rand['status']}@{rand['current_step']}")
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
