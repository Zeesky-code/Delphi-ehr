"""
Forecasting AUROC split by what the labs already showed at the cutoff: does FiLM only read off values
that are already diagnostic, or does it use them below the threshold too?

For each target, every forecasting patient (case or control, same cutoffs as forecast_eval.py) falls
into one group, by the defining lab before (and at) the cutoff:
  A  diagnostic        the lab already meets the diagnostic threshold
  B  not diagnostic    measured, but below the threshold. If FiLM still beats the baseline here, it uses
                       values beyond "diagnosis already visible in the labs"
  C  never measured    no value to use: FiLM should do no better than the baseline (negative control)

Thresholds (fixed before looking at the results):
  diabetes      HbA1c >= 6.5 %
  kidney        eGFR < 60 mL/min/1.73m2, CKD-EPI 2021 (creatinine, age, sex; no race term)
  anaemia       haemoglobin < 12 g/dL (women), < 13 g/dL (men), WHO
  hypertension  systolic blood pressure >= 140 mmHg
--rule latest (default) uses the last value before the cutoff; --rule any counts a patient as diagnostic
if any earlier value met the threshold (stricter about "already visible"). A single value is not a
clinical diagnosis (which needs repeated or confirmed measurements), so group A means "a diagnostic
value was on record", not "the patient had the disease".

With --split_never, group C is split further into C1 (other lab or vital values before the cutoff, which
FiLM can still use, e.g. glucose for diabetes) and C2 (no values at all: the strict negative control).
This split was added after seeing the first results, so it is exploratory.

Lab status comes from the true values, also for the shuffled-values model (it describes the patient).
Reads the forecast_<split>_<ckpt>.npz files from forecast_eval.py, so no model is run. Prints a table and
writes <runs_dir>/forecast_by_lab_status_<split>_<ckpt>_<rule>.json.

Usage (from the repo root):
    python3 scripts/forecast_by_lab_status.py runs/film-mimic4 --split test [--rule any] [--split_never]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from compare_forecasts import paired_delong  # noqa: E402  (scripts/ is on sys.path when run as a script)
from forecast_eval import auroc, choose_cutoffs, lab_raw_ids, target_raw_ids  # noqa: E402
from utils import get_p2i  # noqa: E402

FEMALE_RAW, MALE_RAW = 1, 2
# target: (lab name as in the dataset's labels.csv, before the unit)
TARGET_LAB = {'diabetes': 'HbA1c', 'kidney': 'Creatinine', 'anaemia': 'Hemoglobin', 'hypertension': 'Systolic_BP'}
GROUPS = ['A diagnostic', 'B not diagnostic', 'C never measured']
GROUPS_SPLIT = GROUPS[:2] + ['C1 other values only', 'C2 no values at all']
MIN_PER_CLASS = 10  # fewer cases or controls in a group: no AUROC


def egfr_ckd_epi_2021(creatinine, age_years, female):
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = creatinine / kappa
    return (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age_years * np.where(female, 1.012, 1.0))


def is_diagnostic(target, values, age_years, female):
    if target == 'diabetes':
        return values >= 6.5
    if target == 'kidney':
        return egfr_ckd_epi_2021(values, age_years, female) < 60
    if target == 'anaemia':
        return values < np.where(female, 12.0, 13.0)
    if target == 'hypertension':
        return values >= 140
    raise ValueError(target)


def lab_status(data, values, p2i, patients, rows, target, lab_raw, rule, split_never=False):
    """
    Group index per patient, from the target's lab up to the cutoff row: 0 = A, 1 = B, 2 = C, or with
    split_never 2 = C1 (other values present) and 3 = C2 (no values at all).
    """
    status = np.full(len(patients), 2)
    for k, (p, row) in enumerate(zip(patients, rows)):
        start = p2i[p][0]
        hist = data[start:row + 1]
        is_lab = (hist[:, 2] == lab_raw) & (values[start:row + 1] > 0)
        if not is_lab.any():
            if split_never and not (values[start:row + 1] > 0).any():
                status[k] = 3
            continue
        female = bool((hist[:, 2] == FEMALE_RAW).any())
        v = values[start:row + 1][is_lab].astype(np.float64)
        ages = hist[is_lab, 1] / 365.25
        if rule == 'latest':
            v, ages = v[-1:], ages[-1:]
        status[k] = 0 if is_diagnostic(target, v, ages, female).any() else 1
    return status


def fmt(xs, signed=False):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '-'
    s = f'{np.mean(xs):+.3f}' if signed else f'{np.mean(xs):.3f}'
    return s + (f' ±{np.std(xs):.3f}' if len(xs) > 1 else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs_dir')
    parser.add_argument('--split', default='test')
    parser.add_argument('--ckpt', default='last')
    parser.add_argument('--rule', default='latest', choices=['latest', 'any'])
    parser.add_argument('--data', default='data/mimic4_labs')
    parser.add_argument('--targets', default='config/forecast_targets.json')
    parser.add_argument('--variants', nargs='*', help='default: every variant with forecast results')
    parser.add_argument('--split_never', action='store_true',
                        help='split "never measured" into "other values only" and "no values at all"')
    args = parser.parse_args()

    # the same cutoffs as forecast_eval.py: its settings are stored in each result file
    name = f'forecast_{args.split}_{args.ckpt}'
    runs = {}  # variant -> seed -> npz
    meta = None
    for path in sorted(Path(args.runs_dir).glob(f'*/seed*/{name}.json')):
        js = json.loads(path.read_text())
        settings = (js['seed'], js['gaps'], js['window_days'], js['washout_days'])
        if meta is None:
            meta = settings
        assert settings == meta, f'{path} was made with different settings'
        runs.setdefault(path.parent.parent.name, {})[path.parent.name] = np.load(path.with_suffix('.npz'))
    if 'baseline' not in runs:
        sys.exit(f'no baseline {name}.json under {args.runs_dir}')
    seed, gaps, window_days, washout_days = meta
    variants = ['baseline'] + (args.variants or [v for v in runs if v != 'baseline'])

    data = np.fromfile(os.path.join(args.data, f'{args.split}.bin'), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    values = np.fromfile(os.path.join(args.data, f'{args.split}_values.bin'), dtype=np.float32)
    p2i = get_p2i(data)
    labs = lab_raw_ids(args.data)
    targets = target_raw_ids(args.targets)

    results = []
    header = ['target', 'gap', 'group', 'cases/controls'] + \
             [c for v in variants for c in ([v] if v == 'baseline' else [v, f'Δ {v}', 'CI excl. 0'])]
    table = [header]
    for t_i, (target, raw_ids) in enumerate(targets.items()):
        if target not in TARGET_LAB:
            print(f'skipping {target}: no defining lab')
            continue
        for gap in gaps:
            rng = np.random.default_rng([seed, t_i, int(gap)])
            patients, rows, labels, _ = choose_cutoffs(data, p2i, raw_ids, gap, window_days, washout_days, rng)
            tag = f'{target}_{int(gap)}'
            for seeds in runs.values():
                for arr in seeds.values():
                    assert np.array_equal(arr[f'{tag}_patient'], patients), \
                        f'{tag}: cutoffs differ from forecast_eval.py (other data, split or targets file?)'
            status = lab_status(data, values, p2i, patients, rows, target, labs[TARGET_LAB[target]], args.rule,
                                args.split_never)
            for g, group in enumerate(GROUPS_SPLIT if args.split_never else GROUPS):
                m = status == g
                n1, n0 = int(labels[m].sum()), int((1 - labels[m]).sum())
                row = [target, f'{int(gap)}d', group, f'{n1}/{n0}']
                res = {'target': target, 'gap_days': gap, 'group': group, 'n_cases': n1, 'n_controls': n0}
                ok = n1 >= MIN_PER_CLASS and n0 >= MIN_PER_CLASS
                for v in variants:
                    aucs, deltas, excl = [], [], 0
                    for s, arr in runs.get(v, {}).items():
                        lab, sc = arr[f'{tag}_label'][m], arr[f'{tag}_score'][m]
                        if not ok:
                            continue
                        aucs.append(auroc(sc[lab == 1], sc[lab == 0]))
                        if v != 'baseline' and s in runs['baseline']:
                            d, se = paired_delong(lab, runs['baseline'][s][f'{tag}_score'][m], sc)
                            deltas.append(d)
                            excl += abs(d) > 1.96 * se
                    res[v] = {'auc': aucs, 'delta': deltas, 'ci_excludes_0': excl}
                    row += [fmt(aucs)] if v == 'baseline' else \
                           [fmt(aucs), fmt(deltas, signed=True), f'{excl}/{len(deltas)}' if deltas else '-']
                results.append(res)
                table.append(row)

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for i, r in enumerate(table):
        print('  '.join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            print('  '.join('-' * w for w in widths))
    print(f'\nGroups by {"the latest value" if args.rule == "latest" else "any value"} of the defining lab up to '
          f'the cutoff. AUROC: mean ± sd over seeds; Δ: variant minus baseline, paired by seed. '
          f'"-": fewer than {MIN_PER_CLASS} cases or controls.')
    out = Path(args.runs_dir) / f'forecast_by_lab_status_{args.split}_{args.ckpt}_{args.rule}{"_split" if args.split_never else ""}.json'
    out.write_text(json.dumps({'split': args.split, 'ckpt': args.ckpt, 'rule': args.rule, 'results': results},
                              indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
