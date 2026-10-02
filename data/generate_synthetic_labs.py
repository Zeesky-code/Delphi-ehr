"""
Generate synthetic lab test data extending the existing Delphi synthetic dataset.

This script:
1. Loads the existing synthetic train/val data
2. Defines 10 common lab test types as new tokens after the existing labels
3. For each patient, simulates sparse clinic visits from age 40, each measuring a random subset of labs
4. Correlates lab values with disease events where medically relevant. Values start drifting in
   the years BEFORE a diagnosis (and before death), so a lab value carries information about future
   events that the token sequence alone does not.
5. Saves extended data to a new directory: data/ukb_simulated_data_with_labs/
6. Prints a signal check: how well each linked lab value separates measurements taken shortly
   before a diagnosis from those in patients never diagnosed

Token ids: the .bin files store raw ids, and get_batch() shifts every id by +1 so that 0 can be
padding. Raw id k therefore corresponds to labels.csv entry k+1 (0-based). The original data uses
raw ids up to 1268 (Death, entry 1269), so the lab tokens are raw ids 1269..1278 (entries 1270..1279)
and model ids 1270..1279.

Output format:
  - train.bin / val.bin   : uint32 (patient_id, age_in_days, token_id) — same as original + new lab tokens
  - train_values.bin / val_values.bin : float32, one value per row (0.0 for non-lab tokens)
  - labels.csv            : extended label file with lab test names appended
  - lab_linked_tokens.json : model token ids of the labs and of the diseases each lab group anticipates
"""

import json
import numpy as np
import os
import pickle
import re
import shutil
from pathlib import Path

np.random.seed(42)

# ─── Lab Test Definitions ───────────────────────────────────────────────────────
# Keys are raw token ids (see module docstring).
# Each lab test: (name, unit, normal_mean, normal_std, min_val, max_val, prob_measured_per_visit)
LAB_TESTS = {
    1269: ("HbA1c",           "%",      5.5,   0.5,   3.5,  15.0,  0.2),
    1270: ("Glucose_fasting", "mmol/L", 5.0,   0.6,   2.5,  25.0,  0.2),
    1271: ("Cholesterol_total","mmol/L", 5.2,   1.0,   2.0,  12.0,  0.2),
    1272: ("HDL",             "mmol/L",  1.5,   0.4,   0.5,   3.5,  0.2),
    1273: ("LDL",             "mmol/L",  3.0,   0.9,   0.5,   8.0,  0.2),
    1274: ("Systolic_BP",     "mmHg",  120.0,  12.0,  80.0, 220.0,  0.5),
    1275: ("Creatinine",      "umol/L", 80.0,  15.0,  30.0, 600.0,  0.2),
    1276: ("ALT",             "U/L",    25.0,  10.0,   5.0, 300.0,  0.2),
    1277: ("Hemoglobin",      "g/dL",   14.0,   1.5,   5.0,  20.0,  0.2),
    1278: ("CRP",             "mg/L",    2.0,   2.0,   0.1, 200.0,  0.2),
}

# Clinic visits start at this age and follow with exponential gaps of this mean.
# Labs are kept sparse (~15 per patient) so they don't crowd disease events out of the context.
VISIT_START_AGE_YEARS = 40.0
VISIT_MEAN_GAP_YEARS = 6.0

# Raw token ids of the original data that are not diseases (no event, sex, lifestyle)
FIRST_DISEASE_RAW_TOKEN = 12

# A patient's values for each lab sit at a personal baseline, offset from the population mean by
# N(0, PATIENT_OFFSET_SD * lab std). This makes values vary between people for reasons unrelated
# to disease, so an abnormal value is a noisy signal rather than a perfect one.
PATIENT_OFFSET_SD = 0.5

# Before a diagnosis the shift ramps up linearly over the group's lead_years, reaching
# PRODROME_FRACTION of the full shift at diagnosis; it then reaches the full shift 2 years later.
PRODROME_FRACTION = 0.5

# Disease groups and the labs they move.
# keywords: ICD-10 codes (exact) or text matched at word starts; labels: exact label matches
# lead_years: how long before diagnosis the values start drifting
# effects: {lab raw token: (full shift mean, full shift std)}
DISEASE_LAB_CORRELATIONS = {
    # E10-E14 Diabetes → elevated HbA1c and Glucose
    "diabetes": {
        "keywords": ["diabetes", "E10", "E11", "E12", "E13", "E14"],
        "lead_years": 5.0,
        "effects": {
            1269: (3.0, 1.5),   # HbA1c: +3% mean shift
            1270: (4.0, 2.0),   # Glucose: +4 mmol/L
        }
    },
    # I10-I15 Hypertension → elevated BP
    "hypertension": {
        "keywords": ["hypertension", "hypertensive", "I10", "I11", "I12", "I13"],
        "lead_years": 5.0,
        "effects": {
            1274: (25.0, 10.0),  # Systolic BP: +25 mmHg
        }
    },
    # E78 Hyperlipidemia → elevated cholesterol
    "hyperlipidemia": {
        "keywords": ["hyperlipid", "cholesterol", "E78"],
        "lead_years": 5.0,
        "effects": {
            1271: (2.0, 0.8),   # Total cholesterol: +2
            1273: (1.5, 0.6),   # LDL: +1.5
            1272: (-0.3, 0.15), # HDL: -0.3 (lower is worse)
        }
    },
    # Liver diseases → elevated ALT
    "liver": {
        "keywords": ["liver", "hepat", "K70", "K71", "K72", "K73", "K74", "K75", "K76"],
        "lead_years": 3.0,
        "effects": {
            1276: (50.0, 30.0),  # ALT: +50
        }
    },
    # Kidney diseases → elevated creatinine
    "kidney": {
        "keywords": ["renal failure", "kidney disease", "N17", "N18", "N19"],
        "lead_years": 3.0,
        "effects": {
            1275: (100.0, 50.0),  # Creatinine: +100
        }
    },
    # Anaemia → low hemoglobin
    "anaemia": {
        "keywords": ["anaemia", "anemia", "D50", "D51", "D52", "D53"],
        "lead_years": 2.0,
        "effects": {
            1277: (-3.0, 1.0),  # Hemoglobin: -3
        }
    },
    # Inflammatory conditions → elevated CRP
    "inflammation": {
        "keywords": ["arthritis", "inflammatory", "crohn", "colitis", "M05", "M06", "K50", "K51"],
        "lead_years": 1.0,
        "effects": {
            1278: (20.0, 15.0),  # CRP: +20
        }
    },
    # Declining health before death → lower haemoglobin, higher CRP and creatinine
    "death": {
        "labels": ["Death"],
        "lead_years": 3.0,
        "effects": {
            1277: (-1.5, 0.8),   # Hemoglobin: -1.5
            1278: (8.0, 6.0),    # CRP: +8
            1275: (25.0, 15.0),  # Creatinine: +25
        }
    },
}


def load_labels(labels_path):
    """Load the labels file (one label per line, no header)."""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def find_disease_tokens(labels, keywords=(), exact_labels=()):
    """
    Find raw token IDs whose labels match any of the keywords or exact labels.

    ICD-10 code keywords (e.g. "E11") must equal the label's code; text keywords match the
    start of a word, case-insensitive (so "renal" does not match "adrenal").
    """
    code_re = re.compile(r"^[A-Z]\d\d$")
    matching = set()
    for idx, label in enumerate(labels):
        raw_token = idx - 1  # labels.csv entry k is raw token k-1 (get_batch shifts by +1)
        if raw_token < FIRST_DISEASE_RAW_TOKEN:
            continue
        if label in exact_labels:
            matching.add(raw_token)
            continue
        code = label.split()[0] if label else ""
        for kw in keywords:
            if code_re.match(kw):
                hit = code == kw
            else:
                hit = re.search(r"\b" + re.escape(kw.lower()), label.lower()) is not None
            if hit:
                matching.add(raw_token)
                break
    return matching


def find_group_tokens(labels):
    """Map each correlation group to the raw token ids of its diseases."""
    return {
        group: find_disease_tokens(labels, info.get("keywords", ()), info.get("labels", ()))
        for group, info in DISEASE_LAB_CORRELATIONS.items()
    }


def build_correlation_map(labels, lab_token_offset=0):
    """
    Build a mapping: disease_token_id → {lab_token_id: (shift_mean, shift_std, lead_years)}.
    """
    corr_map = {}
    for group_name, matching_tokens in find_group_tokens(labels).items():
        group_info = DISEASE_LAB_CORRELATIONS[group_name]
        remapped_effects = {
            lab_token + lab_token_offset: (*effect, group_info["lead_years"])
            for lab_token, effect in group_info["effects"].items()
        }
        for tok in matching_tokens:
            if tok not in corr_map:
                corr_map[tok] = {}
            corr_map[tok].update(remapped_effects)
    return corr_map


def generate_labs_for_patient(patient_data, corr_map, rng, lab_tests=LAB_TESTS):
    """
    Generate lab test events for a single patient.

    Args:
        patient_data: (N, 3) array of (patient_id, age_in_days, token_id)
        corr_map: disease_token → lab effects mapping
        rng: numpy random generator

    Returns:
        lab_events: (M, 3) uint32 array of new lab token rows
        lab_values: (M,) float32 array of numeric values
    """
    pid = patient_data[0, 0]
    ages = patient_data[:, 1].astype(np.float64)
    tokens = patient_data[:, 2]

    min_age = ages.min()
    max_age = ages.max()

    # Find which diseases this patient has and when they were first recorded
    patient_diseases = {}
    for age, tok in zip(ages, tokens):
        if tok >= FIRST_DISEASE_RAW_TOKEN:  # Skip no event, sex, lifestyle tokens
            patient_diseases.setdefault(tok, age)

    # Personal baseline per lab
    patient_offsets = {
        lab_token: rng.normal(0, PATIENT_OFFSET_SD * definition[3])
        for lab_token, definition in lab_tests.items()
    }

    lab_events = []
    lab_values = []

    # Simulate clinic visits strictly inside the observed record, so no lab falls on or after
    # the last event (often Death).
    visit_ages = []
    t = max(min_age, VISIT_START_AGE_YEARS * 365.25) + rng.exponential(VISIT_MEAN_GAP_YEARS * 365.25)
    while t < max_age - 1:
        visit_ages.append(int(t))
        t += rng.exponential(VISIT_MEAN_GAP_YEARS * 365.25)

    for m_age in visit_ages:
        for lab_token, (name, unit, mean, std, min_v, max_v, p_measure) in lab_tests.items():
            if rng.random() >= p_measure:
                continue

            # Base value: normal distribution around the patient's personal baseline
            value = rng.normal(mean, std) + patient_offsets[lab_token]

            # Age-related drift (many lab values trend upward with age)
            age_years = m_age / 365.25
            if name in ("HbA1c", "Glucose_fasting"):  # trend up slightly
                value += (age_years - 50) * 0.02
            elif name == "Systolic_BP":  # BP trends up
                value += (age_years - 50) * 0.3
            elif name == "Creatinine":  # Creatinine trends up
                value += (age_years - 50) * 0.5

            # Disease correlations: shift the value in the lead-up to a correlated diagnosis
            # (or death) and after it
            for disease_tok, disease_age in patient_diseases.items():
                if disease_tok in corr_map and lab_token in corr_map[disease_tok]:
                    shift_mean, shift_std, lead_years = corr_map[disease_tok][lab_token]
                    years_after = (m_age - disease_age) / 365.25
                    if years_after >= 0:
                        ramp = PRODROME_FRACTION + (1 - PRODROME_FRACTION) * min(1.0, years_after / 2.0)
                    elif -years_after < lead_years:
                        ramp = PRODROME_FRACTION * (1 + years_after / lead_years)
                    else:
                        continue
                    value += rng.normal(shift_mean * ramp, shift_std * 0.5 * ramp)

            # Clamp to realistic range
            value = np.clip(value, min_v, max_v)

            lab_events.append([pid, m_age, lab_token])
            lab_values.append(value)

    if len(lab_events) == 0:
        return np.empty((0, 3), dtype=np.uint32), np.empty(0, dtype=np.float32)

    return np.array(lab_events, dtype=np.uint32), np.array(lab_values, dtype=np.float32)


def process_dataset(data, corr_map, rng, lab_tests=LAB_TESTS):
    """
    Process an entire dataset: add lab events and generate numeric values.

    Returns:
        extended_data: (N+M, 3) uint32 — original + lab token rows, sorted per patient
        values: (N+M,) float32 — 0.0 for original tokens, actual values for lab tokens
    """
    # Find patient boundaries
    patient_ids = np.unique(data[:, 0])
    print(f"  Processing {len(patient_ids)} patients...")

    all_rows = []
    all_values = []

    for i, pid in enumerate(patient_ids):
        if (i + 1) % 2000 == 0:
            print(f"    Patient {i+1}/{len(patient_ids)}...")

        mask = data[:, 0] == pid
        patient_data = data[mask]

        # Original data: values are 0.0
        orig_values = np.zeros(len(patient_data), dtype=np.float32)

        # Generate lab data
        lab_events, lab_values = generate_labs_for_patient(patient_data, corr_map, rng, lab_tests)

        # Merge
        if len(lab_events) > 0:
            combined = np.vstack([patient_data, lab_events])
            combined_values = np.concatenate([orig_values, lab_values])
        else:
            combined = patient_data
            combined_values = orig_values

        # Sort by age within patient
        sort_idx = np.argsort(combined[:, 1], kind='stable')
        combined = combined[sort_idx]
        combined_values = combined_values[sort_idx]

        all_rows.append(combined)
        all_values.append(combined_values)

    extended_data = np.vstack(all_rows)
    values = np.concatenate(all_values)
    return extended_data, values


def rank_auc(pos, neg):
    """Probability that a random positive scores above a random negative (Mann-Whitney AUC)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def lead_signal_report(data, values, group_tokens, lab_token_offset, lab_names):
    """
    For each group and linked lab, compare measurements taken within the lead window before the
    patient's first diagnosis (positives) against measurements from never-diagnosed patients
    (negatives). AUC 0.5 means the value carries no warning; it is oriented by the shift's sign.
    """
    starts = np.r_[0, np.flatnonzero(np.diff(data[:, 0].astype(np.int64))) + 1, len(data)]
    print(f"\n  {'group':15s} {'lab':18s} {'lead':>5s} {'n_pos':>6s} {'n_neg':>7s} {'AUC':>5s}")
    for group, info in DISEASE_LAB_CORRELATIONS.items():
        tokens = np.array(sorted(group_tokens[group]))
        for lab, (shift_mean, _) in info["effects"].items():
            lab_tok = lab + lab_token_offset
            lead_days = info["lead_years"] * 365.25
            pos, neg = [], []
            for a, b in zip(starts[:-1], starts[1:]):
                tok, age, val = data[a:b, 2], data[a:b, 1].astype(np.float64), values[a:b]
                is_lab = tok == lab_tok
                if not is_lab.any():
                    continue
                dx = np.isin(tok, tokens)
                if not dx.any():
                    neg.extend(val[is_lab])
                else:
                    gap = age[dx].min() - age[is_lab]
                    pos.extend(val[is_lab][(gap > 0) & (gap <= lead_days)])
            sign = 1 if shift_mean > 0 else -1
            auc = rank_auc(sign * np.array(pos), sign * np.array(neg))
            print(f"  {group:15s} {lab_names[lab_tok]:18s} {info['lead_years']:4.0f}y "
                  f"{len(pos):6d} {len(neg):7d} {auc:5.2f}")


def main():
    src_dir = Path(__file__).parent / "ukb_simulated_data"
    dst_dir = Path(__file__).parent / "ukb_simulated_data_with_labs"
    dst_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)

    # Load labels and build correlation map
    labels = load_labels(src_dir / "labels.csv")
    # labels.csv entry k is raw token k-1, so the first new entry (index len(labels)) is raw len(labels)-1
    lab_token_offset = (len(labels) - 1) - min(LAB_TESTS)
    lab_tests = {
        lab_token + lab_token_offset: definition
        for lab_token, definition in LAB_TESTS.items()
    }
    corr_map = build_correlation_map(labels, lab_token_offset)
    group_tokens = find_group_tokens(labels)
    print(f"Found disease-lab correlations for {len(corr_map)} disease tokens")

    # Model token ids (raw + 1) of the labs and of each group's diseases, for targeted evaluation
    with open(dst_dir / "lab_linked_tokens.json", "w") as f:
        json.dump({
            "lab_tokens": sorted(int(t) + 1 for t in lab_tests),
            "groups": {g: sorted(int(t) + 1 for t in toks) for g, toks in group_tokens.items()},
        }, f, indent=1)

    # Extend labels with lab test names
    extended_labels = labels.copy()
    for lab_token in sorted(lab_tests.keys()):
        name, unit, *_ = lab_tests[lab_token]
        extended_labels.append(f"{name} ({unit})")

    # Save extended labels
    with open(dst_dir / "labels.csv", 'w') as f:
        for label in extended_labels:
            f.write(label + "\n")
    print(f"Extended vocab: {len(labels)} → {len(extended_labels)} tokens")

    # Process train and val
    for split in ["train", "val"]:
        print(f"\nProcessing {split}...")
        data = np.fromfile(src_dir / f"{split}.bin", dtype=np.uint32).reshape(-1, 3)
        print(f"  Original: {len(data)} records")

        extended_data, values = process_dataset(data, corr_map, rng, lab_tests)
        print(f"  Extended: {len(extended_data)} records (+{len(extended_data) - len(data)} lab events)")
        print(f"  Lab values range: {values[values > 0].min():.1f} - {values[values > 0].max():.1f}")

        # Save
        extended_data.astype(np.uint32).tofile(dst_dir / f"{split}.bin")
        values.astype(np.float32).tofile(dst_dir / f"{split}_values.bin")

        # Stats
        is_lab = np.isin(extended_data[:, 2], list(lab_tests))
        n_lab = np.sum(is_lab)
        n_orig = np.sum(~is_lab)
        print(f"  Original tokens: {n_orig}, Lab tokens: {n_lab} ({n_lab / len(extended_data):.0%} of rows)")
        print(f"  Avg lab events/patient: {n_lab / len(np.unique(extended_data[:, 0])):.1f}")
        if split == "train":
            print("  Signal check (lab value before first diagnosis vs never diagnosed):")
            lab_names = {t: d[0] for t, d in lab_tests.items()}
            lead_signal_report(extended_data, values, group_tokens, lab_token_offset, lab_names)

    # meta.pkl: same structure as the original, extended to the new vocabulary
    with open(src_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    meta["vocab_size"] = len(extended_labels)
    meta["itos"] = {i: i for i in range(len(extended_labels))}
    meta["stoi"] = {i: i for i in range(len(extended_labels))}
    with open(dst_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Copy other necessary files
    for fname in ["icd10_codes_mod.tsv", "fields.txt"]:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)

    # Print summary of lab value distributions
    print("\n" + "="*60)
    print("LAB VALUE SUMMARY")
    print("="*60)
    data = np.fromfile(dst_dir / "train.bin", dtype=np.uint32).reshape(-1, 3)
    values = np.fromfile(dst_dir / "train_values.bin", dtype=np.float32)
    for lab_token, (name, unit, mean, std, min_v, max_v, _) in lab_tests.items():
        mask = data[:, 2] == lab_token
        if mask.sum() > 0:
            v = values[mask]
            print(f"  {name:20s} ({unit:6s}): n={mask.sum():5d}, "
                  f"mean={v.mean():7.1f}, std={v.std():6.1f}, "
                  f"range=[{v.min():6.1f}, {v.max():7.1f}]")

    print(f"\nDone! Output saved to: {dst_dir}")
    print(f"  {dst_dir}/train.bin          — extended token data")
    print(f"  {dst_dir}/train_values.bin   — parallel numeric values")
    print(f"  {dst_dir}/val.bin            — extended token data")
    print(f"  {dst_dir}/val_values.bin     — parallel numeric values")
    print(f"  {dst_dir}/labels.csv         — extended label file")
    print(f"  {dst_dir}/lab_linked_tokens.json — lab and lab-linked disease token ids")


if __name__ == "__main__":
    main()
