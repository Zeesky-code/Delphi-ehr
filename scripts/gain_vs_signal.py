"""
Does FiLM's gain track how much warning signal the lab values carry?

For each FiLM variant and lab group (diabetes, kidney, death, ...), compares the variant's
validation CE on the group's diseases with the baseline's, paired by seed (gain = baseline CE -
variant CE, so positive = better), and sets it against the group's signal: the AUC of its
strongest lab for an upcoming first diagnosis (from the dataset's signal check).

A shuffled-values control (`<variant>_shuffled`) is drawn in the same colour with open markers:
if FiLM uses the values, the filled points rise with signal and the open ones stay near zero.

Writes <out>.csv (all numbers) and <out>.png (the figure).

Usage:
    python3 scripts/gain_vs_signal.py RUNS_DIR [--data data/mimic4_labs] [--variants attn_pre qkv]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mimic'))

# reference categorical palette, fixed order (one hue per base variant)
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#dcdbd6'


def load_runs(runs_dir):
    runs = {}
    for path in sorted(Path(runs_dir).glob('*/seed*/summary.json')):
        s = json.loads(path.read_text())
        runs.setdefault(path.parent.parent.name, {})[s['seed']] = s
    return runs


def load_signal(data_dir):
    """Strongest lab AUC per group, from <data>/signal_check.csv (computed and saved if missing)."""
    path = os.path.join(data_dir, 'signal_check.csv')
    if not os.path.exists(path):
        import build_mimic4 as b
        print(f'computing signal check from {data_dir} (saved to {path})...')
        b.signal_table(b.events_from_dataset(data_dir), b.load_labels(b.DELPHI_LABELS)).to_csv(path, index=False)
    signal = pd.read_csv(path).dropna(subset=['auc'])  # groups without measurements before a diagnosis
    best = signal.loc[signal.groupby('group')['auc'].idxmax()]
    return best.set_index('group')[['lab', 'auc', 'n_pos']].rename(
        columns={'lab': 'strongest_lab', 'auc': 'signal_auc', 'n_pos': 'n_pos_strongest_lab'})


def gains(runs, variants, groups):
    base = runs['baseline']
    rows = []
    for variant in variants:
        for group in groups:
            key = f'full_val_ce_{group}'
            per_seed = [base[seed][key] - s[key] for seed, s in runs[variant].items()
                        if seed in base and s.get(key) is not None and base[seed].get(key) is not None]
            if not per_seed:
                continue
            rows.append({'variant': variant, 'group': group, 'gain_mean': float(np.mean(per_seed)),
                         'gain_std': float(np.std(per_seed)) if len(per_seed) > 1 else np.nan,
                         'n_seeds': len(per_seed),
                         'n_val_targets': next(iter(runs[variant].values())).get(f'full_val_n_{group}')})
    return pd.DataFrame(rows)


def plot(table, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # groups along x in order of signal strength, so "gain tracks signal" reads left to right
    order = table.groupby('group')['signal_auc'].first().sort_values()
    position = {g: i for i, g in enumerate(order.index)}
    bases = list(dict.fromkeys(re.sub('_shuffled$', '', v) for v in table['variant']))
    series = [(b, v, filled) for b in bases for v, filled in ((b, True), (f'{b}_shuffled', False))
              if (table['variant'] == v).any()]
    width = 0.7 / max(len(series), 1)
    for k, (base, variant, filled) in enumerate(series):
        color = COLORS[bases.index(base) % len(COLORS)]
        part = table[table['variant'] == variant]
        x = part['group'].map(position) + (k - (len(series) - 1) / 2) * width
        ax.errorbar(x, part['gain_mean'], yerr=part['gain_std'].fillna(0), fmt='o', markersize=6,
                    linewidth=1, capsize=2, color=color, zorder=3, markeredgewidth=1.5,
                    markerfacecolor=color if filled else 'white',
                    label=variant + ('' if filled else ' (values shuffled)'))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{g}\n{auc:.2f}' for g, auc in order.items()], fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4)

    seeds = int(table['n_seeds'].max())
    ax.set_xlabel("Disease group, with the AUC of its strongest lab for an upcoming first diagnosis")
    ax.set_ylabel('CE gain over baseline (higher = better)')
    ax.set_title(f'FiLM gain vs lab signal per disease group ({seeds} seed{"s" if seeds > 1 else ""}'
                 f'{", mean ± sd" if seeds > 1 else ""})', fontsize=10, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_png)
    print(f'wrote {out_png}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs_dir')
    parser.add_argument('--data', default='data/mimic4_labs', help='dataset directory (for the signal check)')
    parser.add_argument('--variants', nargs='*', help='variants to include (shuffled controls are added '
                        'automatically when present); default: all')
    parser.add_argument('--out', help='output path without extension (default <runs_dir>/gain_vs_signal)')
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if 'baseline' not in runs:
        sys.exit(f'no baseline runs under {args.runs_dir}')
    variants = [v for v in runs if v != 'baseline']
    if args.variants:
        wanted = set(args.variants) | {f'{v}_shuffled' for v in args.variants}
        variants = [v for v in variants if v in wanted]

    signal = load_signal(args.data)
    table = gains(runs, variants, list(signal.index)).join(signal, on='group')
    out = args.out or os.path.join(args.runs_dir, 'gain_vs_signal')
    table.to_csv(f'{out}.csv', index=False)
    print(table.sort_values(['variant', 'signal_auc']).to_string(index=False, float_format='%.3f'))

    # correlation of gain with signal per variant (few points: a description, not a test)
    for variant, part in table.groupby('variant'):
        if len(part) >= 3:
            rho = part[['signal_auc', 'gain_mean']].corr(method='spearman').iloc[0, 1]
            print(f'{variant:28s} Spearman(signal, gain) over {len(part)} groups = {rho:+.2f}')
    plot(table, f'{out}.png')


if __name__ == '__main__':
    main()
