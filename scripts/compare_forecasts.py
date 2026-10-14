"""
Compare forecasting AUROCs across variants (forecast_<split>_<ckpt>.json/.npz from forecast_eval.py).

Per target and gap: each variant's AUROC (mean ± sd over seeds), and its difference from the baseline
paired by seed. Every model is scored on the same patients and cutoffs, so the difference is tested
with the paired DeLong method; the last column counts the seeds whose 95% CI for the difference
excludes zero. With three seeds this describes consistency, it is not a significance claim.

Usage:
    python3 scripts/compare_forecasts.py runs/film-mimic4 [--variants attn_pre attn_pre_shuffled] [--split test]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluate_auc import fastDeLong  # noqa: E402


def paired_delong(labels, score_a, score_b):
    """AUROC(b) - AUROC(a) and its standard error, for two models scored on the same samples."""
    order = np.argsort(-labels, kind='stable')
    aucs, cov = fastDeLong(np.vstack([score_a, score_b])[:, order], int(labels.sum()))
    return float(aucs[1] - aucs[0]), float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))


def fmt(xs, digits=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '-'
    return f'{np.mean(xs):.{digits}f}' + (f' ±{np.std(xs):.{digits}f}' if len(xs) > 1 else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs_dir')
    parser.add_argument('--variants', nargs='*', help='default: every variant with forecast results')
    parser.add_argument('--split', default='test')
    parser.add_argument('--ckpt', default='last')
    args = parser.parse_args()

    name = f'forecast_{args.split}_{args.ckpt}'
    runs = {}  # variant -> seed -> (json, npz)
    for path in sorted(Path(args.runs_dir).glob(f'*/seed*/{name}.json')):
        seed = path.parent.name
        runs.setdefault(path.parent.parent.name, {})[seed] = (json.loads(path.read_text()),
                                                               np.load(path.with_suffix('.npz')))
    if 'baseline' not in runs:
        sys.exit(f'no baseline {name}.json under {args.runs_dir} (run forecast_eval.py on the baseline runs)')
    variants = args.variants or [v for v in runs if v != 'baseline']
    base = runs['baseline']

    header = ['target', 'gap', 'cases/controls', 'baseline'] + \
             [c for v in variants for c in (v, f'Δ {v}', 'CI excl. 0')]
    rows = [header]
    for res in next(iter(base.values()))[0]['results']:
        target, gap = res['target'], res['gap_days']
        tag = f'{target}_{int(gap)}'
        row = [target, f'{int(gap)}d', f"{res['n_cases']}/{res['n_controls']}",
               fmt([r[0]['results'][i].get('auc') for r in base.values()
                    for i, x in enumerate(r[0]['results']) if x['target'] == target and x['gap_days'] == gap])]
        for v in variants:
            aucs, deltas, excl, n = [], [], 0, 0
            for seed, (js, arr) in runs.get(v, {}).items():
                r = next(x for x in js['results'] if x['target'] == target and x['gap_days'] == gap)
                aucs.append(r.get('auc'))
                if seed in base and r.get('auc') is not None:
                    b_arr = base[seed][1]
                    assert np.array_equal(b_arr[f'{tag}_patient'], arr[f'{tag}_patient']), \
                        'models were scored on different cutoffs'
                    d, se = paired_delong(arr[f'{tag}_label'], b_arr[f'{tag}_score'], arr[f'{tag}_score'])
                    deltas.append(d)
                    excl += abs(d) > 1.96 * se
                    n += 1
            row += [fmt(aucs), fmt(deltas) if deltas else '-', f'{excl}/{n}' if n else '-']
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for i, r in enumerate(rows):
        print('  '.join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            print('  '.join('-' * w for w in widths))
    print('\nAUROC: mean ± sd over seeds. Δ: variant minus baseline, paired by seed (positive = better).')


if __name__ == '__main__':
    main()
