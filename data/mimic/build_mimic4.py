"""
Build a Delphi dataset with numeric lab values from MIMIC-IV (BigQuery), in the same format as
data/ukb_simulated_data_with_labs, so train.py, the FiLM sweep and evaluate_auc.py run unchanged.

Cohort and timeline (ICD-10 only):
  - patients with at least one admission coded in ICD-10
  - each patient's timeline starts at the discharge of their last ICD-9-coded admission (if any),
    so no stays with dropped diagnoses fall inside the record. this is a design choice for simplicity
  - diagnoses: first occurrence of each 3-character ICD-10 code, at the discharge time of the
    admission it was coded in (codes are assigned at discharge), mapped to Delphi's vocabulary;
    codes outside it (mostly R, S-T, V-Y, Z chapters) are dropped, as in Delphi
  - sex at age 0; Death at the date of death (never before the patient's last event)
  - valued tokens: the LAB_PANEL below, from labevents (blood labs) and omr (BMI, blood pressure).
    Implausible values are dropped. To keep sequences short, each is kept at most once per
    admission (first value) and once per outpatient_months outside admissions.
  - ages in days from an approximate birth date (anchor_year - anchor_age, 1 July)

Token ids follow Delphi's labels.csv (raw id k = labels.csv entry k+1; get_batch adds 1), with the
valued tokens appended after Death. Patients are split 80/10/10 into train/val/test.

"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from generate_synthetic_labs import find_group_tokens, load_labels, rank_auc  # noqa: E402

DELPHI_LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ukb_simulated_data', 'labels.csv')

# name, source table, key (labevents itemid or omr result_name), unit, plausible range
LAB_PANEL = [
    ('HbA1c',             'lab', 50852,           '%',     3,    20),
    ('Glucose',           'lab', 50931,           'mg/dL', 10,   2000),
    ('Creatinine',        'lab', 50912,           'mg/dL', 0.1,  25),
    ('Urea_nitrogen',     'lab', 51006,           'mg/dL', 1,    300),
    ('Hemoglobin',        'lab', 51222,           'g/dL',  2,    25),
    ('ALT',               'lab', 50861,           'IU/L',  1,    10000),
    ('AST',               'lab', 50878,           'IU/L',  1,    20000),
    ('Cholesterol_total', 'lab', 50907,           'mg/dL', 30,   1000),
    ('HDL',               'lab', 50904,           'mg/dL', 2,    250),
    ('LDL',               'lab', 50905,           'mg/dL', 1,    600),
    ('Triglycerides',     'lab', 51000,           'mg/dL', 10,   10000),
    ('CRP',               'lab', 50889,           'mg/L',  0.01, 700),
    ('Albumin',           'lab', 50862,           'g/dL',  0.5,  7),
    ('Platelets',         'lab', 51265,           'K/uL',  1,    2000),
    ('WBC',               'lab', 51301,           'K/uL',  0.05, 500),
    ('BMI',               'omr', 'BMI (kg/m2)',    'kg/m2', 10,   100),
    ('Systolic_BP',       'omr', 'Blood Pressure', 'mmHg',  50,   300),
]

# Labs expected to move before each group's first diagnosis (+1 higher, -1 lower), for the
# signal check. Groups and their diseases come from data/generate_synthetic_labs.py.
SIGNAL_LABS = {
    'diabetes':       {'HbA1c': 1, 'Glucose': 1},
    'hypertension':   {'Systolic_BP': 1},
    'hyperlipidemia': {'Cholesterol_total': 1, 'LDL': 1, 'HDL': -1, 'Triglycerides': 1},
    'liver':          {'ALT': 1, 'AST': 1},
    'kidney':         {'Creatinine': 1, 'Urea_nitrogen': 1},
    'anaemia':        {'Hemoglobin': -1},
    'inflammation':   {'CRP': 1},
    'death':          {'Albumin': -1, 'Hemoglobin': -1, 'CRP': 1, 'Urea_nitrogen': 1},
}
SIGNAL_LEAD_DAYS = 2 * 365.25

FEMALE_RAW, MALE_RAW = 1, 2  # labels.csv entries 2 (Female) and 3 (Male)
ORDER_SEX, ORDER_VALUE, ORDER_DX, ORDER_DEATH = 0, 1, 2, 3  # tie-break for events on the same day


# ─── BigQuery extraction ──────────────────────────────────────────────────────────

def extraction_queries(hosp, outpatient_months):
    adm, pat, dx = f"`{hosp}.admissions`", f"`{hosp}.patients`", f"`{hosp}.diagnoses_icd`"
    lab, omr = f"`{hosp}.labevents`", f"`{hosp}.omr`"
    cohort = f"""
    cohort AS (
      SELECT a.subject_id, MAX(IF(d.icd_version = 9, a.dischtime, NULL)) AS window_start
      FROM {adm} a JOIN {dx} d USING (subject_id, hadm_id)
      GROUP BY a.subject_id
      HAVING LOGICAL_OR(d.icd_version = 10))"""

    def period(t):  # outpatient bucket: year and n-month period
        return f"FORMAT('%d-%d', EXTRACT(YEAR FROM {t}), DIV(EXTRACT(MONTH FROM {t}) - 1, {outpatient_months}))"

    lab_rows = [x for x in LAB_PANEL if x[1] == 'lab']
    omr_rows = [x for x in LAB_PANEL if x[1] == 'omr']
    lab_ranges = ' OR '.join(f'(l.itemid = {key} AND l.valuenum BETWEEN {lo} AND {hi})'
                             for _, _, key, _, lo, hi in lab_rows)
    omr_ranges = ' OR '.join(f"(key = '{key}' AND value BETWEEN {lo} AND {hi})"
                             for _, _, key, _, lo, hi in omr_rows)
    omr_names = ', '.join(f"'{key}'" for _, _, key, *_ in omr_rows)

    return {
        'patients': f"""
        WITH {cohort}
        SELECT p.subject_id, p.gender, p.anchor_age, p.anchor_year, p.dod, c.window_start
        FROM {pat} p JOIN cohort c USING (subject_id)""",

        'diagnoses': f"""
        WITH {cohort}
        SELECT d.subject_id, SUBSTR(d.icd_code, 1, 3) AS code3, MIN(a.dischtime) AS time
        FROM {dx} d
        JOIN {adm} a USING (subject_id, hadm_id)
        JOIN cohort c USING (subject_id)
        WHERE d.icd_version = 10 AND (c.window_start IS NULL OR a.admittime >= c.window_start)
        GROUP BY 1, 2""",

        'labs': f"""
        WITH {cohort},
        l AS (
          SELECT l.subject_id, l.hadm_id, l.itemid AS key, l.charttime AS time, l.valuenum AS value
          FROM {lab} l JOIN cohort c USING (subject_id)
          WHERE ({lab_ranges}) AND (c.window_start IS NULL OR l.charttime >= c.window_start)),
        ranked AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY subject_id, key, IFNULL(CAST(hadm_id AS STRING), {period('time')})
            ORDER BY time) AS rn
          FROM l)
        SELECT subject_id, CAST(key AS STRING) AS key, time, value FROM ranked WHERE rn = 1""",

        'omr': f"""
        WITH {cohort},
        o AS (
          SELECT o.subject_id, o.result_name AS key, DATETIME(o.chartdate) AS time,
                 SAFE_CAST(SPLIT(o.result_value, '/')[SAFE_OFFSET(0)] AS FLOAT64) AS value
          FROM {omr} o JOIN cohort c USING (subject_id)
          WHERE o.result_name IN ({omr_names})
            AND (c.window_start IS NULL OR DATETIME(o.chartdate) >= c.window_start)),
        valid AS (SELECT * FROM o WHERE {omr_ranges}),
        ranked AS (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id, key, {period('time')} ORDER BY time) AS rn
          FROM valid)
        SELECT subject_id, key, time, value FROM ranked WHERE rn = 1""",
    }


def extract(project, hosp, outpatient_months, max_gb, cache_dir):
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * 1e9))
    os.makedirs(cache_dir, exist_ok=True)
    tables = {}
    for name, sql in extraction_queries(hosp, outpatient_months).items():
        print(f"querying {name}...")
        job = client.query(sql, job_config=job_config)
        df = job.to_dataframe()
        print(f"  {len(df):,} rows ({(job.total_bytes_billed or 0) / 1e9:.1f} GB billed)")
        df.to_parquet(os.path.join(cache_dir, f'{name}.parquet'), index=False)
        tables[name] = df
    return tables


def load_cache(cache_dir):
    return {name: pd.read_parquet(os.path.join(cache_dir, f'{name}.parquet'))
            for name in ['patients', 'diagnoses', 'labs', 'omr']}


# ─── Building the token sequences ─────────────────────────────────────────────────

def build_events(tables, labels):
    """Turn the extracted tables into one event table: subject_id, age (days), token (raw), value."""
    patients = tables['patients'].copy()
    birth_year = (patients['anchor_year'] - patients['anchor_age']).astype(int)
    patients['birth'] = pd.to_datetime(birth_year.astype(str) + '-07-01')
    birth = patients.set_index('subject_id')['birth']

    def age_days(df):
        times = pd.to_datetime(df['time'])  # DATETIME, TIMESTAMP (tz-aware) or DATE columns
        if times.dt.tz is not None:
            times = times.dt.tz_localize(None)
        return (times - df['subject_id'].map(birth)).dt.total_seconds() / 86400

    events = []

    sex = pd.DataFrame({
        'subject_id': patients['subject_id'],
        'age': 0.0,
        'token': np.where(patients['gender'] == 'F', FEMALE_RAW, MALE_RAW),
        'value': 0.0, 'order': ORDER_SEX})
    events.append(sex)

    # diagnoses → Delphi raw token ids (label entry k is raw token k-1); skip non-disease labels
    code_to_raw = {}
    for idx, label in enumerate(labels):
        raw = idx - 1
        if 12 <= raw and label != 'Death' and label:
            code_to_raw.setdefault(label.split()[0], raw)
    dx = tables['diagnoses']
    dx_raw = dx['code3'].map(code_to_raw)
    mapped = dx_raw.notna()
    events.append(pd.DataFrame({
        'subject_id': dx.loc[mapped, 'subject_id'], 'age': age_days(dx[mapped]),
        'token': dx_raw[mapped].astype(int), 'value': 0.0, 'order': ORDER_DX}))
    coverage = {
        'diagnosis_rows': len(dx), 'mapped_rows': int(mapped.sum()),
        'top_unmapped': dx.loc[~mapped, 'code3'].value_counts().head(15).to_dict()}

    # valued tokens appended after Death
    first_value_raw = len(labels) - 1
    key_to_raw = {str(key): first_value_raw + i for i, (_, _, key, *_) in enumerate(LAB_PANEL)}
    for name in ['labs', 'omr']:
        df = tables[name]
        events.append(pd.DataFrame({
            'subject_id': df['subject_id'], 'age': age_days(df),
            'token': df['key'].astype(str).map(key_to_raw).astype(int),
            'value': df['value'].astype(float), 'order': ORDER_VALUE}))

    events = pd.concat(events, ignore_index=True)

    # death at the date of death, but never before the patient's last recorded event
    died = patients[patients['dod'].notna()]
    dod_age = age_days(pd.DataFrame({'subject_id': died['subject_id'], 'time': died['dod']}))
    last_age = events.groupby('subject_id')['age'].max()
    death_age = np.maximum(dod_age.to_numpy(), died['subject_id'].map(last_age).to_numpy())
    death_raw = labels.index('Death') - 1
    events = pd.concat([events, pd.DataFrame({
        'subject_id': died['subject_id'].to_numpy(), 'age': death_age,
        'token': death_raw, 'value': 0.0, 'order': ORDER_DEATH})], ignore_index=True)

    events['age'] = np.floor(events['age'].clip(lower=0)).astype(np.int64)
    events = events.sort_values(['subject_id', 'age', 'order'], kind='stable').reset_index(drop=True)
    return events, coverage


def split_patients(subject_ids, seed, fractions=(0.8, 0.1, 0.1)):
    ids = np.array(sorted(subject_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train, n_val = int(fractions[0] * len(ids)), int(fractions[1] * len(ids))
    return {'train': set(ids[:n_train]), 'val': set(ids[n_train:n_train + n_val]),
            'test': set(ids[n_train + n_val:])}


def signal_report(events, labels):
    """
    Per group and linked lab: AUC of values measured within 2 years before the patient's first
    diagnosis in the group (positives) against values from patients never diagnosed in it
    (negatives). Real data is observational, so this mixes the lab's warning signal with
    who gets tested; AUC 0.5 means no separation.
    """
    group_tokens = find_group_tokens(labels)
    first_value_raw = len(labels) - 1
    lab_raw = {name: first_value_raw + i for i, (name, *_) in enumerate(LAB_PANEL)}
    print(f"\n  {'group':15s} {'lab':18s} {'n_pos':>8s} {'n_neg':>9s} {'AUC':>5s}")
    for group, lab_signs in SIGNAL_LABS.items():
        dx = events[events['token'].isin(group_tokens[group])]
        first_dx = dx.groupby('subject_id')['age'].min()
        for lab, sign in lab_signs.items():
            vals = events[events['token'] == lab_raw[lab]]
            dx_age = vals['subject_id'].map(first_dx)
            gap = dx_age - vals['age']
            pos = vals.loc[(gap > 0) & (gap <= SIGNAL_LEAD_DAYS), 'value'].to_numpy()
            neg = vals.loc[dx_age.isna(), 'value'].to_numpy()
            auc = rank_auc(sign * pos, sign * neg)
            print(f"  {group:15s} {lab:18s} {len(pos):8d} {len(neg):9d} {auc:5.2f}")


def write_dataset(events, labels, out_dir, seed, block_size):
    os.makedirs(out_dir, exist_ok=True)
    value_names = [f"{name} ({unit})" for name, _, _, unit, *_ in LAB_PANEL]
    extended_labels = labels + value_names
    with open(os.path.join(out_dir, 'labels.csv'), 'w') as f:
        f.write('\n'.join(extended_labels) + '\n')
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        n = len(extended_labels)
        pickle.dump({'vocab_size': n, 'itos': {i: i for i in range(n)}, 'stoi': {i: i for i in range(n)}}, f)

    first_value_raw = len(labels) - 1
    with open(os.path.join(out_dir, 'lab_linked_tokens.json'), 'w') as f:
        json.dump({
            'lab_tokens': [first_value_raw + i + 1 for i in range(len(LAB_PANEL))],
            'groups': {g: sorted(int(t) + 1 for t in toks) for g, toks in find_group_tokens(labels).items()},
        }, f, indent=1)

    splits = split_patients(events['subject_id'].unique(), seed)
    for split, ids in splits.items():
        part = events[events['subject_id'].isin(ids)]
        rows = np.stack([part['subject_id'].to_numpy(), part['age'].to_numpy(), part['token'].to_numpy()], 1)
        rows.astype(np.uint32).tofile(os.path.join(out_dir, f'{split}.bin'))
        part['value'].to_numpy(np.float32).tofile(os.path.join(out_dir, f'{split}_values.bin'))
        lengths = part.groupby('subject_id').size()
        print(f"  {split:5s}: {len(ids):7,d} patients, {len(part):11,d} events, "
              f"events per patient p50/p90/p99 = {lengths.quantile(.5):.0f}/{lengths.quantile(.9):.0f}/"
              f"{lengths.quantile(.99):.0f}, longer than block_size {block_size}: {(lengths > block_size).mean():.1%}")
    return extended_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', help='your GCP project (billed for the queries)')
    parser.add_argument('--hosp', default='physionet-data.mimiciv_3_1_hosp')
    parser.add_argument('--out_dir', default='data/mimic4_labs')
    parser.add_argument('--outpatient_months', type=int, default=3,
                        help='keep each lab at most once per this many months outside admissions')
    parser.add_argument('--from_cache', action='store_true',
                        help='rebuild from <out_dir>/raw/*.parquet instead of querying BigQuery')
    parser.add_argument('--block_size', type=int, default=256, help='only used to report sequence lengths')
    parser.add_argument('--max_gb', type=float, default=100, help='per-query cap on GB billed')
    parser.add_argument('--seed', type=int, default=42, help='seed for the train/val/test split')
    args = parser.parse_args()

    cache_dir = os.path.join(args.out_dir, 'raw')
    if args.from_cache:
        tables = load_cache(cache_dir)
    else:
        if not args.project:
            parser.error('--project is required unless --from_cache')
        tables = extract(args.project, args.hosp, args.outpatient_months, args.max_gb, cache_dir)

    labels = load_labels(DELPHI_LABELS)
    print("\nbuilding sequences...")
    events, coverage = build_events(tables, labels)
    print(f"  diagnoses mapped to Delphi's vocabulary: {coverage['mapped_rows']:,} of "
          f"{coverage['diagnosis_rows']:,} first-occurrence codes "
          f"({coverage['mapped_rows'] / max(coverage['diagnosis_rows'], 1):.0%})")
    print(f"  most common unmapped codes: {coverage['top_unmapped']}")
    counts = events['order'].value_counts()
    print(f"  events: {len(events):,} (valued tokens {counts.get(ORDER_VALUE, 0) / len(events):.0%}, "
          f"diagnoses {counts.get(ORDER_DX, 0) / len(events):.0%}, deaths {counts.get(ORDER_DEATH, 0):,})")

    print("\nwriting splits...")
    extended_labels = write_dataset(events, labels, args.out_dir, args.seed, args.block_size)

    print("\nsignal check (all patients):")
    signal_report(events, labels)
    print(f"\nvocab_size = {len(extended_labels)}; valued tokens are model ids "
          f"{len(labels)}-{len(extended_labels) - 1}. Output in {args.out_dir}/")


if __name__ == '__main__':
    main()
