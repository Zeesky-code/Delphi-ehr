"""
Write a small random dataset in the build_mimic4.py output format, for crash-testing the training,
sweep and evaluation code without MIMIC access. The data is random: it says nothing about results.

Usage: python3 data/mimic/make_fake_mimic4.py OUT_DIR [--patients 800] [--icd9_gem 2018_I9gem.txt]
With --icd9_gem, diagnoses are built as with build_mimic4.py --icd9 map (some patients ICD-9 only).
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


def fake_tables(n_patients, rng, icd9_codes=None):
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
    patients['has_icd10'] = True
    if icd9_codes is not None:  # full codes of both versions, as the --icd9 map extraction returns them
        patients['window_start'] = pd.NaT
        only9 = set(rng.choice(patients['subject_id'], n_patients // 3, replace=False))
        patients['has_icd10'] = ~patients['subject_id'].isin(only9)
        is9 = diagnoses['subject_id'].isin(only9) | (rng.random(len(diagnoses)) < .3)
        diagnoses = diagnoses.assign(icd_code=diagnoses['code3'] + '9', icd_version=10)
        diagnoses.loc[is9, 'icd_code'] = rng.choice(icd9_codes, int(is9.sum()))
        diagnoses.loc[is9, 'icd_version'] = 9
        diagnoses = diagnoses.drop(columns='code3')
    return labels, {
        'patients': patients, 'diagnoses': diagnoses,
        'labs': valued([x for x in b.LAB_PANEL if x[1] == 'lab'], 40 * n_patients),
        'omr': valued([x for x in b.LAB_PANEL if x[1] == 'omr'], 8 * n_patients)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_dir')
    parser.add_argument('--patients', type=int, default=800)
    parser.add_argument('--icd9_gem', help='CMS GEM file: also make ICD-9 diagnoses and map them')
    args = parser.parse_args()
    gem = b.load_icd9_gem(args.icd9_gem) if args.icd9_gem else None
    labels, tables = fake_tables(args.patients, np.random.default_rng(0), None if gem is None else sorted(gem))
    train_only = ()
    if gem is not None:
        tables['diagnoses'], coverage = b.map_icd9(tables['diagnoses'], gem)
        print(coverage)
        train_only = set(tables['patients'].loc[~tables['patients']['has_icd10'], 'subject_id'])
    events, _ = b.build_events(tables, labels)
    b.write_dataset(events, labels, args.out_dir, seed=42, block_size=256, train_only=train_only)
    b.signal_table(events, labels).to_csv(os.path.join(args.out_dir, 'signal_check.csv'), index=False)


if __name__ == '__main__':
    main()
