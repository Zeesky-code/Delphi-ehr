"""
Compare FiLM variant runs written by scripts/run_film_variants.sh.

Reads every <runs_dir>/<variant>/seed*/summary.json and prints, per variant, the mean (± std over
seeds) of the end-of-run evaluation of the best checkpoint on the whole validation set:

  val loss    CE + dt over all scored targets
  CE linked   CE on targets that are lab-linked diseases, the diseases a lab value can warn about.
              This is where FiLM should help; the overall loss dilutes it.
  Δ columns   difference from the 'baseline' variant (negative = better)

A second table breaks CE down by lab group (diabetes, kidney, death, ...).

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
    return float(np.mean(xs)), (float(np.std(xs)) if len(xs) > 1 else None)


def fmt(m, s=None, digits=3):
    if m is None:
        return '-'
    return f'{m:.{digits}f}' + (f' ±{s:.{digits}f}' if s is not None else '')


def print_table(table):
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for i, row in enumerate(table):
        print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            print('  '.join('-' * w for w in widths))


def main():
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/film')
    groups = {}
    for path in sorted(runs_dir.glob('*/seed*/summary.json')):
        groups.setdefault(path.parent.parent.name, []).append(json.loads(path.read_text()))
    if not groups:
        sys.exit(f'No summary.json files under {runs_dir}')
    if not any('full_val_loss' in r for runs in groups.values() for r in runs):
        sys.exit('These summaries predate the full-validation evaluation; rerun with the current train.py')

    def stat(runs, key):
        return mean_std([r.get(key) for r in runs])

    rows = {name: {'runs': runs, 'loss': stat(runs, 'full_val_loss'),
                   'linked': stat(runs, 'full_val_ce_linked')} for name, runs in groups.items()}
    base = rows.get('baseline')

    def delta(r, key):
        if base is None or r[key][0] is None or base[key][0] is None:
            return '-'
        return f'{r[key][0] - base[key][0]:+.4f}'

    order = sorted(rows, key=lambda n: (n != 'baseline', rows[n]['linked'][0] or rows[n]['loss'][0] or 1e18))

    table = [['variant', 'seeds', 'val loss', 'Δ loss', 'CE linked', 'Δ linked', 'val CE', 'val dt',
              'best iter', 'params', 'FiLM params', 'ms/iter']]
    for name in order:
        r, runs = rows[name], rows[name]['runs']
        table.append([
            name, str(len(runs)), fmt(*r['loss']), delta(r, 'loss'), fmt(*r['linked']), delta(r, 'linked'),
            fmt(*stat(runs, 'full_val_ce')), fmt(*stat(runs, 'full_val_dt')),
            fmt(stat(runs, 'best_iter')[0], digits=0),
            f"{runs[0]['n_params'] / 1e6:.2f}M", f"{runs[0]['n_film_params'] / 1e6:.3f}M",
            fmt(stat(runs, 'ms_per_iter')[0], digits=0),
        ])
    print_table(table)

    # CE per lab group
    lab_groups = sorted({k[len('full_val_ce_'):] for runs in groups.values() for r in runs
                         for k in r if k.startswith('full_val_ce_')} - {'linked'})
    if lab_groups:
        first = rows[order[0]]['runs'][0]
        print('\nCE on each group\'s diseases (number of validation targets in brackets)')
        table = [['variant'] + [f"{g} ({first.get(f'full_val_n_{g}', '?')})" for g in lab_groups]]
        for name in order:
            table.append([name] + [fmt(stat(rows[name]['runs'], f'full_val_ce_{g}')[0]) for g in lab_groups])
        print_table(table)


if __name__ == '__main__':
    main()
