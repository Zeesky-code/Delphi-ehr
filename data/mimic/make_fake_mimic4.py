"""
Write a small random dataset in the build_mimic4.py output format, for crash-testing the training,
sweep and evaluation code without MIMIC access. The data is random: it says nothing about results.

Usage: python3 data/mimic/make_fake_mimic4.py OUT_DIR [--patients 800]
       python3 train.py config/train_delphi_mimic4_film.py --dataset=$PWD/OUT_DIR --device=cpu --max_iters=4
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_mimic4 as b  # noqa: E402


def fake_tables(n_patients, rng):
    labels = b.load_labels(b.DELPHI_LABELS)
    codes = [label.split()[0] for label in labels[13:1269]][:300] + ['Z87', 'Y92', 'R07']  # some unmapped
    patients = pd.DataFrame({
        'subject_id': np.arange(10_000_000, 10_000_000 + n_patients),
        'gender': rng.choice(['F', 'M'], n_patients),
        'anchor_age': rng.integers(18, 91, n_patients),
        'anchor_year': rng.integers(2110, 2180, n_patients)})
    patients['window_start'] = [pd.Timestamp(f'{y + 1}-03-01') if rng.random() < .3 else pd.NaT
                                for y in patients['anchor_year']]
    patients['dod'] = [dt.date(y + 5, 6, 1) if rng.random() < .2 else None for y in patients['anchor_year']]
    year = patients.set_index('subject_id')['anchor_year']

    def times(subjects):
        return pd.to_datetime([f'{year[s] + 1 + rng.integers(0, 4)}-{rng.integers(1, 13):02d}-15 10:00'
                               for s in subjects])

    def valued(rows, n):
        i = rng.integers(0, len(rows), n)
        subjects = rng.choice(patients['subject_id'], n)
        lo, hi = np.array([r[4] for r in rows])[i], np.array([r[5] for r in rows])[i]
        return pd.DataFrame({'subject_id': subjects, 'key': [str(rows[k][2]) for k in i],
                             'time': times(subjects), 'value': np.exp(rng.uniform(np.log(lo), np.log(hi)))})

    n_dx = 8 * n_patients
    subjects = rng.choice(patients['subject_id'], n_dx)
    diagnoses = pd.DataFrame({'subject_id': subjects, 'code3': rng.choice(codes, n_dx), 'time': times(subjects)})
    diagnoses = diagnoses.groupby(['subject_id', 'code3'], as_index=False)['time'].min()
    return labels, {
        'patients': patients, 'diagnoses': diagnoses,
        'labs': valued([x for x in b.LAB_PANEL if x[1] == 'lab'], 40 * n_patients),
        'omr': valued([x for x in b.LAB_PANEL if x[1] == 'omr'], 8 * n_patients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_dir')
    parser.add_argument('--patients', type=int, default=800)
    args = parser.parse_args()
    labels, tables = fake_tables(args.patients, np.random.default_rng(0))
    events, _ = b.build_events(tables, labels)
    b.write_dataset(events, labels, args.out_dir, seed=42, block_size=256)
    b.signal_table(events, labels).to_csv(os.path.join(args.out_dir, 'signal_check.csv'), index=False)


if __name__ == '__main__':
    main()
