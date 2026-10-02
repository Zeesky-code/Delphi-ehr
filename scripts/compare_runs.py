"""
Compare FiLM variant runs written by scripts/run_film_variants.sh.

Reads every <runs_dir>/<variant>/seed*/summary.json and prints one row per variant with the mean
(± std over seeds) of the validation losses, plus parameter counts and time per iteration.
The last column is the change in best validation loss against the 'baseline' variant.

Usage: python3 scripts/compare_runs.py [runs_dir]   (default runs/film)
"""

import json
import sys
from pathlib import Path

import numpy as np


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs)) if len(xs) > 1 else None


def fmt(m, s=None, digits=3):
    if m is None:
        return '-'
    return f'{m:.{digits}f}' + (f' ±{s:.{digits}f}' if s is not None else '')


def main():
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/film')
    groups = {}
    for path in sorted(runs_dir.glob('*/seed*/summary.json')):
        groups.setdefault(path.parent.parent.name, []).append(json.loads(path.read_text()))
    if not groups:
        sys.exit(f'No summary.json files under {runs_dir}')

    rows = {}
    for name, runs in groups.items():
        rows[name] = {
            'seeds': len(runs),
            'best': mean_std([r['best_val_loss'] for r in runs]),
            'ce': mean_std([r['final_val_loss_ce'] for r in runs]),
            'dt': mean_std([r['final_val_loss_dt'] for r in runs]),
            'params': runs[0]['n_params'],
            'film_params': runs[0]['n_film_params'],
            'ms': mean_std([r['ms_per_iter'] for r in runs])[0],
        }
    base = rows.get('baseline', {}).get('best', (None,))[0]

    header = ['variant', 'seeds', 'best val loss', 'val CE', 'val dt', 'params', 'FiLM params',
              'ms/iter', 'Δ best vs baseline']
    table = [header]
    for name, r in sorted(rows.items(), key=lambda kv: (kv[0] != 'baseline', kv[1]['best'][0] or 1e18)):
        delta = r['best'][0] - base if base is not None and r['best'][0] is not None else None
        table.append([
            name, str(r['seeds']), fmt(*r['best']), fmt(*r['ce']), fmt(*r['dt']),
            f"{r['params'] / 1e6:.2f}M", f"{r['film_params'] / 1e6:.3f}M",
            fmt(r['ms'], digits=0), fmt(delta) if delta is not None else '-',
        ])

    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for i, row in enumerate(table):
        print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            print('  '.join('-' * w for w in widths))


if __name__ == '__main__':
    main()
