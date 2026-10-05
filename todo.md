# FiLM for lab values: to-do

## Next week

### Controls and repeat runs
- [ ] Add a shuffled-values control: a `film_shuffle_values` option that permutes each lab's
      values across patients during training and evaluation. Same parameters and compute, but no
      information in the values. Add `qkv_shuffled` and `both_numeric_shuffled` to
      `scripts/run_film_variants.sh`.
- [ ] Rerun with 3 seeds (`SEEDS="42 43 44"`): `baseline`, `qkv`, `both_numeric`,
      `both_layers01`, `attn_pre`, plus the shuffled controls.
- [ ] `scripts/compare_runs.py`: show the spread between seeds and paired per-seed deltas against
      the baseline, overall and per group.

### Analysis
- [ ] Fix the metrics before looking at results. Primary: `CE linked`. Secondary: per-group CE
      for every variant. Don't pick the best variant per disease.
- [ ] Gain-vs-signal figure: per group, Δ CE against the number of measurements before diagnosis
      (or the generator's signal-check AUC). Single-seed hint so far: hypertension (most signal)
      improves in 10/11 variants, liver (least signal) gets worse in 10/11.
- [ ] Look into why every FiLM variant has a lower time loss (dt) than the baseline: is it the
      values (e.g. the death prodrome) or extra capacity? The shuffled control answers this.
- [ ] AUC on lab-linked diseases for the final 2–3 models:
      `evaluate_auc.py --diseases lab_linked`.

### Write-up corrections (from the single-seed analysis)
- [ ] Delphi's time loss is an exponential waiting-time likelihood, not a discrete-time survival loss.
- [ ] `qkv` has 3.34M parameters (2.24M base + 1.10M FiLM), not 4.14M.
- [ ] The synthetic data has creatinine, not eGFR. Diabetes is linked only to HbA1c and glucose.
- [ ] Remove the claim that adding labs improved the model; there was no no-labs comparison.
- [ ] Remove "both_layers01 regularises"; adding parameters doesn't regularise.
- [ ] Drop the efficiency-index chart, and start loss axes at a sensible baseline.
- [ ] Present the qkv / early-layer explanations as hypotheses, not findings.

## Later
- [ ] xVal-style input scaling (`emb = f(z) · wte(lab)`) as a zero-parameter baseline against FiLM.
- [ ] Auxiliary head that predicts the next lab value.
- [ ] Move to real UK Biobank labs; consider age/sex-specific normalisation.
- [ ] Repo cleanup: add a `.gitignore`, untrack `__pycache__/*.pyc` and `.DS_Store`, remove
      `data/interpolation_temp_1522.csv` and the empty `Delphi-film-*smoke` folders.


Hello Flavio, this week I read some more papers and started implementing Feature-wise Linear Modulation. For now, I am experimenting with where to place it in the architecture. Because I extended Delphi's synthetic dataset with 10 labs, including HbA1c, glucose, lipids, BP, creatinine, ALT, Hb and CRP. and the idea is to tokenize the test and modulate it based on the lab results. For now, some options of where to place it: attention input, query/key/value, after attention, after the MLP, or both. Also options for scale-only vs shift-only, what the generator sees, which layers get FiLM, and generator width. Every variant starts out exactly matching the baseline, so any difference comes from training.

I have only used one seed but several FiLM variants show lower loss on lab-linked diseases than the baseline Delphi architecture, by about 0.06–0.09. The most consistent pattern is that the gain tracks how much warning signal each disease group has. Hypertension, with the most data, improves in 10 of 11 variants. Liver disease, with very little data, gets worse in 10 of 11.

Next week, I want to add a shuffled-values control to getter a better control. Also, I will read about how gated encoders can be used here. I had a quick chat with Joana and she siad she hasn't tried them yet.


- Synthetic lab data. I extended Delphi's synthetic dataset with 10 labs, including HbA1c, glucose, lipids, BP, creatinine, ALT, Hb and CRP. Along the way I fixed token-id and disease-linkage bugs and reduced labs from 88% to 37% of events. Values now drift before related diagnoses and before death, so they carry information about future events. A built-in check confirms the signal: lab-value AUC for an upcoming diagnosis is 0.55–0.85 depending on the group.
- Evaluation. Each run ends by scoring its best checkpoint on the whole validation set. It reports cross-entropy on the diseases labs can warn about, overall and per group. I also fixed the AUC script so it works with lab values.
- Infrastructure. A sweep script runs 12 variants against a labs-without-values baseline, logs to wandb, and produces a comparison table.

Preliminary results (one seed, not conclusive)
- Several FiLM variants (qkv, both_numeric, both_layers01) show lower loss on lab-linked diseases than the baseline, by about 0.06–0.09.
- The most consistent pattern is that the gain tracks how much warning signal each disease group has. Hypertension, with the most data, improves in 10 of 11 variants. Liver disease, with very little data, gets worse in 10 of 11.
- The differences between variants are small, and with one seed I can't tell them from training noise yet. So I'm not ranking variants.

Next week
1. Add a shuffled-values control: FiLM trained with values randomly permuted across patients, so the values carry no information. This separates "uses the values" from "just has more parameters".
2. Rerun the top 4 variants plus the baseline and controls with 3 seeds, and report gains against the spread between seeds.
3. Make a gain-vs-signal figure per disease group, and compute AUC for the final 2–3 models.
4. Correct a few points in my draft analysis (loss definition, parameter counts, overclaimed conclusions).