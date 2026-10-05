"""
Scope MIMIC-IV or MIMIC-III in BigQuery for a Delphi-style lab-value experiment.

Answers the questions that decide the data design:
  1. overview         patients, admissions, diagnosis and lab rows, deaths
  2. admissions       how many admissions per patient, and over how many years
  3. icd_versions     ICD-9 vs ICD-10 coverage (Delphi's vocabulary is ICD-10)
  4. icd10_codes      3-character ICD-10 codes by patient count, to match against Delphi's labels
  5. top_labs         the 100 lab tests measured in the most patients
  6. candidate_labs   the lab tests matching our synthetic panel (HbA1c, glucose, lipids, ...)
  7. lab_density      lab results per admission, and outside admissions
  8. new_onset        per disease group: patients diagnosed, diagnosed for the first time at a
                      later admission, and of those, how many had a related lab beforehand.
                      This is the real-data version of the generator's signal check.
  9. omr              outpatient measurements such as BMI and blood pressure (MIMIC-IV only)

Each result is printed and saved as CSV in --out_dir. Every query is capped at --max_gb billed.

Colab:
    from google.colab import auth; auth.authenticate_user()
    %run data/mimic/scope_mimic.py --project YOUR_GCP_PROJECT --dataset mimic4
"""

import argparse
import os

import pandas as pd
from google.cloud import bigquery

# Table and column names per dataset. icd_version is a SQL expression (MIMIC-III is all ICD-9).
DATASETS = {
    'mimic4': dict(hosp='physionet-data.mimiciv_3_1_hosp', icd_code='icd_code', icd_version='icd_version'),
    'mimic3': dict(hosp='physionet-data.mimiciii_clinical', icd_code='icd9_code', icd_version='9'),
}

# Disease groups mirroring the synthetic generator. dx: regex on ICD-10 or ICD-9 codes (stored
# without dots); lab: regex on d_labitems.label for the labs expected to warn about the group.
GROUPS = [
    ('diabetes',       r'^(E1[0-4]|250)',                         r'(?i)a1c|^glucose$'),
    ('hypertension',   r'^(I1[0-35]|40[1-5])',                    r'(?i)^creatinine$'),
    ('hyperlipidemia', r'^(E78|272)',                             r'(?i)cholesterol|triglycerides'),
    ('liver',          r'^(K7[0-6]|57[0-3])',                     r'(?i)alanine aminotransferase|asparate aminotransferase|aspartate aminotransferase|bilirubin, total'),
    ('kidney',         r'^(N1[7-9]|58[4-6])',                     r'(?i)^creatinine$|^urea nitrogen$'),
    ('anaemia',        r'^(D5[0-9]|D6[0-4]|28[0-5])',             r'(?i)^hemoglobin$|^hematocrit$'),
    ('inflammation',   r'^(M0[5-6]|K5[01]|714|555|556)',          r'(?i)c-reactive protein'),
]
CANDIDATE_LABS = (r'(?i)a1c|^glucose$|cholesterol|triglycerides|^creatinine$|alanine aminotransferase'
                  r'|^hemoglobin$|c-reactive protein|^urea nitrogen$')


def queries(t, dataset):
    adm, pat, dx = f"`{t['hosp']}.admissions`", f"`{t['hosp']}.patients`", f"`{t['hosp']}.diagnoses_icd`"
    lab, dlab = f"`{t['hosp']}.labevents`", f"`{t['hosp']}.d_labitems`"
    code, version = t['icd_code'], t['icd_version']
    groups_sql = ',\n      '.join(
        f"STRUCT('{g}' AS grp, r'{dx_re}' AS dx_pattern, r'{lab_re}' AS lab_pattern)"
        for g, dx_re, lab_re in GROUPS)

    q = {}
    q['overview'] = f"""
    SELECT
      (SELECT COUNT(DISTINCT subject_id) FROM {pat}) AS patients,
      (SELECT COUNT(*) FROM {adm}) AS admissions,
      (SELECT COUNT(*) FROM {dx}) AS diagnosis_rows,
      (SELECT COUNT(*) FROM {lab}) AS lab_rows,
      (SELECT COUNTIF(hadm_id IS NULL) FROM {lab}) AS lab_rows_outside_admissions,
      (SELECT COUNTIF(dod IS NOT NULL) FROM {pat}) AS patients_with_death_date,
      (SELECT COUNTIF(hospital_expire_flag = 1) FROM {adm}) AS in_hospital_deaths"""

    q['admissions'] = f"""
    WITH per AS (
      SELECT subject_id, COUNT(*) AS n_adm,
             DATE_DIFF(DATE(MAX(admittime)), DATE(MIN(admittime)), DAY) / 365.25 AS span_years
      FROM {adm} GROUP BY subject_id)
    SELECT CASE WHEN n_adm = 1 THEN '1' WHEN n_adm = 2 THEN '2' WHEN n_adm <= 5 THEN '3-5'
                WHEN n_adm <= 10 THEN '6-10' ELSE '11+' END AS admissions_per_patient,
           COUNT(*) AS patients,
           ROUND(APPROX_QUANTILES(span_years, 2)[OFFSET(1)], 2) AS median_span_years,
           ROUND(MAX(span_years), 1) AS max_span_years
    FROM per GROUP BY 1 ORDER BY MIN(n_adm)"""

    q['icd_versions'] = f"""
    WITH d AS (SELECT subject_id, hadm_id, {code} AS icd_code, {version} AS icd_version FROM {dx}),
    per_patient AS (SELECT icd_version, subject_id, COUNT(DISTINCT hadm_id) AS n_adm
                    FROM d GROUP BY 1, 2),
    multi AS (SELECT icd_version, COUNTIF(n_adm >= 2) AS patients_with_2plus_admissions
              FROM per_patient GROUP BY 1),
    base AS (SELECT icd_version, COUNT(*) AS rows_, COUNT(DISTINCT subject_id) AS patients,
                    COUNT(DISTINCT hadm_id) AS admissions,
                    COUNT(DISTINCT SUBSTR(icd_code, 1, 3)) AS distinct_3char_codes
             FROM d GROUP BY 1)
    SELECT * FROM base JOIN multi USING (icd_version) ORDER BY icd_version"""

    q['icd10_codes'] = f"""
    SELECT SUBSTR({code}, 1, 3) AS code3, COUNT(DISTINCT subject_id) AS patients
    FROM {dx} WHERE {version} = 10
    GROUP BY 1 ORDER BY patients DESC"""

    q['top_labs'] = f"""
    SELECT l.itemid, d.label, d.fluid, d.category,
           COUNT(DISTINCT l.subject_id) AS patients, COUNT(*) AS results,
           ROUND(COUNTIF(l.valuenum IS NOT NULL) / COUNT(*), 3) AS frac_numeric,
           ROUND(COUNTIF(l.hadm_id IS NULL) / COUNT(*), 3) AS frac_outside_admissions,
           APPROX_TOP_COUNT(l.valueuom, 1)[SAFE_OFFSET(0)].value AS unit
    FROM {lab} l JOIN {dlab} d USING (itemid)
    GROUP BY 1, 2, 3, 4 ORDER BY patients DESC LIMIT 100"""

    q['candidate_labs'] = f"""
    SELECT l.itemid, d.label, d.fluid,
           COUNT(DISTINCT l.subject_id) AS patients, COUNT(*) AS results,
           ROUND(COUNTIF(l.valuenum IS NOT NULL) / COUNT(*), 3) AS frac_numeric,
           ROUND(COUNTIF(l.hadm_id IS NULL) / COUNT(*), 3) AS frac_outside_admissions,
           APPROX_TOP_COUNT(l.valueuom, 1)[SAFE_OFFSET(0)].value AS unit,
           APPROX_QUANTILES(l.valuenum, 4) AS quartiles
    FROM {lab} l JOIN {dlab} d USING (itemid)
    WHERE REGEXP_CONTAINS(d.label, r'{CANDIDATE_LABS}') AND d.fluid = 'Blood'
    GROUP BY 1, 2, 3 ORDER BY patients DESC"""

    q['lab_density'] = f"""
    WITH per_adm AS (
      SELECT hadm_id, COUNT(*) AS n, COUNT(DISTINCT itemid) AS n_types
      FROM {lab} WHERE hadm_id IS NOT NULL GROUP BY 1),
    per_patient_outside AS (
      SELECT subject_id, COUNT(*) AS n FROM {lab} WHERE hadm_id IS NULL GROUP BY 1)
    SELECT 'results per admission' AS measure, APPROX_QUANTILES(n, 10) AS deciles FROM per_adm
    UNION ALL
    SELECT 'distinct lab types per admission', APPROX_QUANTILES(n_types, 10) FROM per_adm
    UNION ALL
    SELECT 'results outside admissions per patient (patients with any)', APPROX_QUANTILES(n, 10)
    FROM per_patient_outside"""

    q['new_onset'] = f"""
    WITH grps AS (SELECT * FROM UNNEST([
      {groups_sql}])),
    adm_ranked AS (
      SELECT subject_id, hadm_id, admittime,
             ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) AS k
      FROM {adm}),
    first_dx AS (
      SELECT g.grp, d.subject_id, MIN(a.admittime) AS first_time, MIN(a.k) AS first_k
      FROM {dx} d
      JOIN adm_ranked a USING (subject_id, hadm_id)
      JOIN grps g ON REGEXP_CONTAINS(d.{code}, g.dx_pattern)
      GROUP BY 1, 2),
    lab_before AS (
      SELECT DISTINCT f.grp, f.subject_id
      FROM first_dx f
      JOIN grps g USING (grp)
      JOIN {lab} l ON l.subject_id = f.subject_id AND l.charttime < f.first_time
      JOIN {dlab} dl ON dl.itemid = l.itemid
      WHERE f.first_k >= 2 AND l.valuenum IS NOT NULL AND REGEXP_CONTAINS(dl.label, g.lab_pattern)),
    lab_counts AS (SELECT grp, COUNT(*) AS of_which_related_lab_before FROM lab_before GROUP BY 1),
    dx_counts AS (SELECT grp, COUNT(*) AS patients_with_dx,
                         COUNTIF(first_k >= 2) AS first_dx_at_later_admission
                  FROM first_dx GROUP BY 1)
    SELECT dx_counts.*, IFNULL(lab_counts.of_which_related_lab_before, 0) AS of_which_related_lab_before
    FROM dx_counts LEFT JOIN lab_counts USING (grp)
    ORDER BY patients_with_dx DESC"""

    if dataset == 'mimic4':
        q['omr'] = f"""
        SELECT result_name, COUNT(DISTINCT subject_id) AS patients, COUNT(*) AS results
        FROM `{t['hosp']}.omr` GROUP BY 1 ORDER BY patients DESC"""
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, help='your GCP project (billed for the queries)')
    parser.add_argument('--dataset', choices=DATASETS, default='mimic4')
    parser.add_argument('--hosp', help='override the hosp dataset, e.g. physionet-data.mimiciv_2_2_hosp')
    parser.add_argument('--only', nargs='*', help='run only these queries, e.g. --only overview new_onset')
    parser.add_argument('--max_gb', type=float, default=50, help='per-query cap on GB billed')
    parser.add_argument('--out_dir', default='mimic_scoping')
    args = parser.parse_args()

    tables = dict(DATASETS[args.dataset])
    if args.hosp:
        tables['hosp'] = args.hosp
    client = bigquery.Client(project=args.project)

    project = tables['hosp'].split('.')[0]
    try:
        names = sorted(d.dataset_id for d in client.list_datasets(project))
        print(f"{project} datasets you can see: {', '.join(n for n in names if 'mimic' in n)}\n")
    except Exception as e:  # listing needs extra permissions; querying may still work
        print(f"(could not list {project} datasets: {e.__class__.__name__})\n")

    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=int(args.max_gb * 1e9))
    pd.set_option('display.width', 200, 'display.max_columns', 20, 'display.max_rows', 120)

    for name, sql in queries(tables, args.dataset).items():
        if args.only and name not in args.only:
            continue
        print(f"=== {name}")
        try:
            job = client.query(sql, job_config=job_config)
            df = job.to_dataframe()
        except Exception as e:
            print(f"failed: {e}\n")
            continue
        df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
        print(df.head(40 if name != 'top_labs' else 100).to_string(index=False))
        print(f"({(job.total_bytes_billed or 0) / 1e9:.1f} GB billed)\n")
    print(f"CSVs saved in {out_dir}/")


if __name__ == '__main__':
    main()
