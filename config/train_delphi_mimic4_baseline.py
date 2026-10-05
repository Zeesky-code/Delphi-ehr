
import time

# MIMIC-IV with lab tokens but no values; built by data/mimic/build_mimic4.py
out_dir = 'Delphi-mimic4-baseline'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi-mimic4'
wandb_run_name = 'mimic4-baseline-' + str(time.time())

dataset = 'mimic4_labs'
batch_size = 128
block_size = 256  # hospital records are longer than UK Biobank's; check the build's length report
data_fraction = 1.0

# Same architecture as the synthetic runs
n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.0
weight_decay = 2e-1
vocab_size = 1287  # Delphi's 1270 labels + 17 valued tokens (15 labs, BMI, systolic BP)

# ~180k training patients: 10k iterations of 128 is about 7 passes over the data
learning_rate = 2e-3
max_iters = 10000
lr_decay_iters = 10000
min_lr = 2e-4
beta2 = 0.99

warmup_iters = 1000

# Ignore padding, sex and lifestyle tokens (as in Delphi) and the valued tokens (model ids
# 1270-1286): they are inputs only, so the model is never trained to predict or generate them
ignore_tokens = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + list(range(1270, 1287))
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
