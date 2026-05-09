from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path('/jizhicfs/wuyanning/cluster_data_selection')
OUTPUTS = ROOT / 'outputs'
RUN_ROOTS = {
    'm32': OUTPUTS / 'exp3_hidden_l2rm1pc_real_m32_20260429_1708_s42',
    'm48': OUTPUTS / 'exp3_hidden_l2rm1pc_real_m48_20260429_1708_s42',
    'm24': OUTPUTS / 'exp3_hidden_l2rm1pc_real_m24_20260429_1708_s42',
    'm64': OUTPUTS / 'exp3_hidden_l2rm1pc_real_m64_20260429_1708_s42',
}
RUN_NAMES = ('hidden_utility_meta', 'hidden_random_meta')
BASE_WEIGHTED = 1.8435
POLL_SECONDS = 300
STATUS_JSON = OUTPUTS / 'exp3_sweep_watch_status.json'
SUMMARY_MD = OUTPUTS / 'exp3_sweep_final_summary.md'
SUMMARY_JSON = OUTPUTS / 'exp3_sweep_final_summary.json'
LOG_PREFIX = '[exp3-watch]'

STEP_RE = re.compile(r'\[step=(\d{6})\] step=(\d+)/300')
EVAL_RE = re.compile(r'\[step=(\d{6})\] eval: math=(\S+)\s+math_ppl=(\S+)\s+reading=(\S+)\s+reading_ppl=(\S+)\s+code=(\S+)\s+code_ppl=(\S+)\s+weighted=(\S+)\s+weighted_ppl=(\S+)')
DONE_RE = re.compile(r'\[step=000300\] Training complete\.')
PMP_RE = re.compile(r'\[PMP\] weights: min=(\S+), max=(\S+), entropy=(\S+), alive=(\d+)/(\d+), dropped=(\d+)')

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
    done = bool(DONE_RE.search(text))
    status = 'done' if done else 'running'
    current_step = int(steps[-1][1]) if steps else None
    if evals:
        last = evals[-1]
        last_eval_step = int(last[0])
        math = float(last[1])
        reading = float(last[3])
        code = float(last[5])
        weighted = float(last[7])
    else:
        last_eval_step = None
        math = reading = code = weighted = None
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
        math=math,
        reading=reading,
        code=code,
        entropy=entropy,
        max_weight=max_weight,
        alive=alive,
    )


def snapshot() -> dict:
    snap = {}
    for tag, root in RUN_ROOTS.items():
        snap[tag] = {}
        for run_name in RUN_NAMES:
            snap[tag][run_name] = parse_run(root / 'runs' / run_name / 'train.log').__dict__
    return snap


def all_done(snap: dict) -> bool:
    for tag in RUN_ROOTS:
        for run_name in RUN_NAMES:
            if snap[tag][run_name]['status'] != 'done':
                return False
    return True


def build_summary(snap: dict) -> tuple[str, dict]:
    rows = []
    best_utility = None
    closest_to_base = None
    for tag in ('m24', 'm32', 'm48', 'm64'):
        util = snap[tag]['hidden_utility_meta']
        rand = snap[tag]['hidden_random_meta']
        util_w = util['weighted']
        rand_w = rand['weighted']
        util_vs_rand = util_w - rand_w
        util_vs_base = util_w - BASE_WEIGHTED
        rows.append({
            'm': int(tag[1:]),
            'utility_weighted': util_w,
            'random_weighted': rand_w,
            'utility_minus_random': round(util_vs_rand, 4),
            'utility_minus_base': round(util_vs_base, 4),
            'utility_math': util['math'],
            'utility_reading': util['reading'],
            'utility_code': util['code'],
            'utility_entropy': util['entropy'],
            'utility_max_weight': util['max_weight'],
            'random_entropy': rand['entropy'],
            'random_max_weight': rand['max_weight'],
        })
        if best_utility is None or util_w < best_utility['utility_weighted']:
            best_utility = rows[-1]
        if closest_to_base is None or abs(util_vs_base) < abs(closest_to_base['utility_minus_base']):
            closest_to_base = rows[-1]
    wins = sum(1 for r in rows if r['utility_minus_random'] < 0)
    summary = {
        'base_weighted': BASE_WEIGHTED,
        'rows': rows,
        'utility_beats_random_count': wins,
        'best_utility_m': best_utility['m'] if best_utility else None,
        'closest_to_base_m': closest_to_base['m'] if closest_to_base else None,
    }
    lines = [
        '### Exp3 m-sweep 自动汇总',
        '',
        f'- **base weighted**: {BASE_WEIGHTED:.4f}',
        f'- **utility 优于 random 的 m 个数**: {wins}/4',
        f'- **utility 最优的 m**: {best_utility["m"] if best_utility else "N/A"}',
        f'- **最接近 base 的 utility m**: {closest_to_base["m"] if closest_to_base else "N/A"}',
        '',
        '| m | utility weighted | random weighted | utility-random | utility-base | utility(math/reading/code) |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for r in rows:
        lines.append(
            f"| {r['m']} | {r['utility_weighted']:.4f} | {r['random_weighted']:.4f} | {r['utility_minus_random']:+.4f} | {r['utility_minus_base']:+.4f} | {r['utility_math']:.4f} / {r['utility_reading']:.4f} / {r['utility_code']:.4f} |"
        )
    lines.extend([
        '',
        '### 自动结论',
        '',
        '- **结论 1**：先看同一 `m` 下 `utility` 是否稳定优于 `random`。',
        '- **结论 2**：再看 `utility` 与 `base` 的差距是否随 `m` 变大而缩小。',
        '- **结论 3**：重点关注最优 `m` 的 `reading/code` 是否继续保持局部优势。',
        '',
    ])
    return '\n'.join(lines), summary


def main() -> None:
    print(f'{LOG_PREFIX} start watching {len(RUN_ROOTS)} sweep dirs', flush=True)
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
            md, js = build_summary(snap)
            SUMMARY_MD.write_text(md, encoding='utf-8')
            SUMMARY_JSON.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{LOG_PREFIX} all runs finished; wrote {SUMMARY_MD.name} and {SUMMARY_JSON.name}', flush=True)
            break
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
