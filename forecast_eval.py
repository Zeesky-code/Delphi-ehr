"""
Forecasting evaluation: does the model predict a future *recorded* diagnosis from the history
available at a cutoff, at least `gap` days ahead?

Design (one cutoff per patient, per target and gap):
  cases     patients whose first recorded code of the target lies at least `gap` days after one of
            their clinical events: the cutoff is the last such event, and the case counts if the
            diagnosis falls within the window (cutoff + gap, cutoff + gap + window_days]
  controls  patients never coded with the target: a time point is drawn uniformly (fixed seed) between
            their first clinical event and `gap` days before their last event, and the cutoff is the
            last clinical event at or before it. Cases and controls are thus both cut at "the last
            event before a moment in time" (mostly the end of an admission); cutting controls at a random
            event instead would mostly land mid-admission, where the model expects diagnoses soon, and
            bias every model's AUROC downwards
  excluded  cases with no event at least `gap` days before the diagnosis or outside the window,
            and (with --washout_days) cases coded within that many days of their record's start,
            when the condition was probably already present

The model sees each patient's events up to the cutoff only: the input is rebuilt from those rows,
so later events cannot leak in (checked on every batch). The score is the model's predicted rate of
the next event being any code of the target (log-sum-exp of their logits) right after the cutoff.
Cutoffs depend only on the data and the seed, so every model is scored on the same patients and
cutoffs, which makes paired comparisons possible (scripts/compare_forecasts.py).

Reported per target and gap: cases, controls, prevalence, AUROC with a DeLong 95% CI, an
age-stratified AUROC (mean over 10-year bands, as Delphi), and the AUROC of age alone for reference.
This predicts a future recorded diagnosis, not disease onset.

Writes <run>/forecast_<split>_<ckpt>.json and .npz (per-patient scores for paired comparisons).

--drop_values LAB [LAB ...] (FiLM runs only) replaces every value of these labs by the training average
(z-score 0) while keeping the tokens, so the model still sees that the test was done but not its result.
Results go to <runs_dir>/<variant>_drop-<LAB>/<seed>/, which the comparison scripts treat as one more
variant: its difference from the full FiLM model is what that lab's values contribute.

Usage:
    python3 forecast_eval.py runs/film-mimic4/*/seed* --split test [--gaps 30 60 365] [--window_days 365]
    python3 forecast_eval.py runs/film-mimic4/attn_pre/seed* --split test --drop_values Glucose
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluate import data_dir_of, load_run, load_split
from evaluate_auc import compute_midrank, get_auc_delong_var
from utils import get_batch, get_p2i

LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ukb_simulated_data', 'labels.csv')
FIRST_CLINICAL_RAW = 12  # raw ids below this are padding, "no event", sex and lifestyle tokens


def target_raw_ids(targets_path):
    """{target: raw token ids} from 3-character ICD-10 codes (labels.csv entry k is raw token k-1)."""
    labels = [line.rstrip('\n') for line in open(LABELS)]
    code_to_raw = {}
    for idx, label in enumerate(labels):
        if label and idx - 1 >= FIRST_CLINICAL_RAW:
            code_to_raw.setdefault(label.split()[0], idx - 1)
    targets = {}
    for name, codes in json.load(open(targets_path)).items():
        if name.startswith('_'):
            continue
        missing = [c for c in codes if c not in code_to_raw]
        if missing:
            raise ValueError(f'{name}: codes not in the vocabulary: {missing}')
        targets[name] = np.array([code_to_raw[c] for c in codes])
    return targets


def lab_raw_ids(data_dir):
    """{lab name: raw token id} from the dataset's labels.csv (line i is model id i, raw id i - 1)."""
    lines = [line.rstrip('\n') for line in open(os.path.join(data_dir, 'labels.csv'))]
    return {line.split(' (')[0]: i - 1 for i, line in enumerate(lines)}


def drop_values(data, values, values_stats, raw_ids):
    """Copy of values with each given lab's values set to its training average (z-score 0)."""
    out = values.copy()
    for raw in raw_ids:
        rows = (data[:, 2] == raw) & (out > 0)
        out[rows] = np.exp(values_stats[raw][0])
    return out


def choose_cutoffs(data, p2i, raw_ids, gap, window_days, washout_days, rng):
    """
    One cutoff row per eligible patient. Returns (patient index, cutoff row, label) arrays, plus counts
    of excluded cases by reason.
    """
    ages, tokens = data[:, 1].astype(np.float64), data[:, 2]
    patients, rows, labels = [], [], []
    excluded = {'no_history_before_gap': 0, 'outside_window': 0, 'washout': 0, 'no_control_cutoff': 0}
    for p, (start, length) in enumerate(p2i):
        a, t = ages[start:start + length], tokens[start:start + length]
        clinical = np.flatnonzero(t >= FIRST_CLINICAL_RAW)
        hit = np.flatnonzero(np.isin(t, raw_ids))
        if len(hit):
            first = a[hit[0]]
            if washout_days and first - a[clinical[0]] < washout_days:
                excluded['washout'] += 1
                continue
            before = clinical[a[clinical] <= first - gap]
            if len(before) == 0:
                excluded['no_history_before_gap'] += 1
                continue
            cut = before[-1]
            if window_days and first - a[cut] > gap + window_days:
                excluded['outside_window'] += 1
                continue
            label = 1
        else:
            if len(clinical) == 0 or a[-1] - gap < a[clinical[0]]:
                excluded['no_control_cutoff'] += 1
                continue
            moment = rng.uniform(a[clinical[0]], a[-1] - gap)
            cut = clinical[a[clinical] <= moment][-1]
            label = 0
        patients.append(p)
        rows.append(start + cut)
        labels.append(label)
    return np.array(patients), np.array(rows), np.array(labels), excluded


def truncated_inputs(data, values, p2i, patients, rows):
    """
    Each selected patient's rows up to and including the cutoff, plus one "no event" row a day later
    so that the cutoff event is the last *input* position. Nothing after the cutoff is copied.
    """
    pieces, vpieces, new_p2i, pos = [], [], [], 0
    for p, row in zip(patients, rows):
        start = p2i[p][0]
        hist = data[start:row + 1]
        dummy = np.array([[hist[0, 0], hist[-1, 1] + 1, 0]], dtype=data.dtype)  # raw 0 = "no event"
        pieces.append(np.vstack([hist, dummy]))
        if values is not None:
            vpieces.append(np.append(values[start:row + 1], 0.0).astype(np.float32))
        new_p2i.append([pos, len(hist) + 1])
        pos += len(hist) + 1
    tdata = np.vstack(pieces)
    tvalues = np.concatenate(vpieces) if values is not None else None
    return tdata, tvalues, np.array(new_p2i)


@torch.no_grad()
def score(model, tdata, tvalues, tp2i, cut_ages, values_stats, model_ids, block_size, batch_size, device,
          no_event_token_rate):
    """Log rate of the next event being any target code, right after each cutoff."""
    out = []
    for start in range(0, len(tp2i), batch_size):
        ix = np.arange(start, min(start + batch_size, len(tp2i)))
        batch = get_batch(ix, tdata, tp2i, select='right', block_size=block_size, device=device,
                          padding='regular', no_event_token_rate=no_event_token_rate, cut_batch=True,
                          values_data=tvalues, values_stats=values_stats)
        X, A, Y, B, V = batch if tvalues is not None else (*batch, None)
        # leakage check: no input event after the cutoff (+1 day: a "no event" pad may sit before the dummy)
        limit = torch.tensor(cut_ages[ix], device=A.device, dtype=A.dtype)[:, None] + 1
        assert bool(((A <= limit) | (X == 0)).all()), 'an input event lies after its cutoff'
        logits, _, _ = model(X, A, Y, B, numeric_values=V, return_attention=False)
        out.append(torch.logsumexp(logits[:, -1, model_ids].float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def auroc(cases, controls):
    """P(case score > control score), ties counting half."""
    n1, n2 = len(cases), len(controls)
    ranks = compute_midrank(np.concatenate([cases, controls]).astype(np.float64))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def age_stratified_auc(scores, labels, ages_years, band=10):
    aucs = []
    for lo in np.arange(np.floor(ages_years.min() / band) * band, ages_years.max() + band, band):
        m = (ages_years >= lo) & (ages_years < lo + band)
        if labels[m].sum() >= 2 and (1 - labels[m]).sum() >= 2:
            aucs.append(auroc(scores[m & (labels == 1)], scores[m & (labels == 0)]))
    return float(np.mean(aucs)) if aucs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--ckpt', default='last', choices=['last', 'best'])
    parser.add_argument('--targets', default='config/forecast_targets.json')
    parser.add_argument('--gaps', type=float, nargs='+', default=[30, 60, 365],
                        help='minimum days between the cutoff and the recorded diagnosis')
    parser.add_argument('--window_days', type=float, default=365,
                        help='a case counts if diagnosed within this many days after the gap (0 = no limit)')
    parser.add_argument('--washout_days', type=float, default=0,
                        help='exclude cases coded within this many days of their first clinical event')
    parser.add_argument('--seed', type=int, default=0, help='for drawing control cutoffs (same for every model)')
    parser.add_argument('--data', help='dataset directory (default: data/<dataset of the run>)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--drop_values', nargs='+', metavar='LAB',
                        help='set these labs\' values to the training average (FiLM runs only), e.g. Glucose')
    args = parser.parse_args()

    targets = target_raw_ids(args.targets)
    cache = {}  # cutoffs depend only on the data, so they are shared by all runs on the same dataset
    for run_dir in args.runs:
        model, checkpoint = load_run(run_dir, args.ckpt, args.device)
        data_dir = data_dir_of(checkpoint, args.data)
        data, values = load_split(data_dir, args.split, checkpoint)
        out_dir = run_dir
        if args.drop_values:
            if values is None:
                print(f'skipping {run_dir}: --drop_values needs a FiLM run')
                continue
            labs = lab_raw_ids(data_dir)
            unknown = [lab for lab in args.drop_values if lab not in labs]
            if unknown:
                raise ValueError(f'labs not in {data_dir}/labels.csv: {unknown}')
            values = drop_values(data, values, {int(k): v for k, v in checkpoint['values_stats'].items()},
                                 [labs[lab] for lab in args.drop_values])
            run = os.path.normpath(run_dir)
            out_dir = os.path.join(os.path.dirname(os.path.dirname(run)),
                                   os.path.basename(os.path.dirname(run)) + '_drop-' + '-'.join(args.drop_values),
                                   os.path.basename(run))
            os.makedirs(out_dir, exist_ok=True)
        p2i = get_p2i(data)
        block_size = checkpoint['model_args']['block_size']
        rate = checkpoint['config'].get('no_event_token_rate', 5)
        results, arrays = [], {}
        for t_i, (name, raw_ids) in enumerate(targets.items()):
            for gap in args.gaps:
                key = (data_dir, name, gap)
                if key not in cache:
                    rng = np.random.default_rng([args.seed, t_i, int(gap)])
                    cache[key] = choose_cutoffs(data, p2i, raw_ids, gap, args.window_days, args.washout_days, rng)
                patients, rows, labels, excluded = cache[key]
                tdata, tvalues, tp2i = truncated_inputs(data, values, p2i, patients, rows)
                cut_ages = data[rows, 1].astype(np.float32)
                s = score(model, tdata, tvalues, tp2i, cut_ages, checkpoint.get('values_stats'), raw_ids + 1,
                          block_size, args.batch_size, args.device, rate)
                n_cases, n_controls = int(labels.sum()), int((1 - labels).sum())
                row = {'target': name, 'gap_days': gap,
                       'window_end_days': gap + args.window_days if args.window_days else None,
                       'n_cases': n_cases, 'n_controls': n_controls,
                       'prevalence': n_cases / max(len(labels), 1), 'excluded': excluded}
                if n_cases >= 2 and n_controls >= 2:
                    a, var = get_auc_delong_var(s[labels == 0], s[labels == 1])
                    half = 1.96 * float(np.sqrt(var))
                    ages_years = cut_ages / 365.25
                    row.update({'auc': float(a), 'auc_ci_low': float(a) - half, 'auc_ci_high': float(a) + half,
                                'auc_age_stratified': age_stratified_auc(s, labels, ages_years),
                                'auc_age_only': auroc(ages_years[labels == 1], ages_years[labels == 0])})
                results.append(row)
                tag = f'{name}_{int(gap)}'
                arrays.update({f'{tag}_patient': patients, f'{tag}_label': labels, f'{tag}_score': s,
                               f'{tag}_age': cut_ages})
                print(f"{out_dir} | {name:13s} gap {int(gap):4d}d: {n_cases:5d} cases / {n_controls:6d} controls, "
                      f"AUROC {row.get('auc', float('nan')):.3f} "
                      f"[{row.get('auc_ci_low', float('nan')):.3f}, {row.get('auc_ci_high', float('nan')):.3f}], "
                      f"age-stratified {row.get('auc_age_stratified') or float('nan'):.3f}, "
                      f"age alone {row.get('auc_age_only', float('nan')):.3f}")
        print('leakage check passed: no input event after its cutoff; '
              'lab values normalised with statistics from the training split (saved in the checkpoint)')
        out = os.path.join(out_dir, f'forecast_{args.split}_{args.ckpt}')
        meta = {'split': args.split, 'ckpt': os.path.basename(checkpoint['path']), 'seed': args.seed,
                'run': run_dir, 'drop_values': args.drop_values,
                'gaps': args.gaps, 'window_days': args.window_days, 'washout_days': args.washout_days,
                'targets': {k: json.load(open(args.targets))[k] for k in targets}}
        with open(out + '.json', 'w') as f:
            json.dump({**meta, 'results': results}, f, indent=2)
        np.savez_compressed(out + '.npz', **arrays)
        print(f'wrote {out}.json and .npz')


if __name__ == '__main__':
    main()
