
import time

# MIMIC-IV with lab values fed through FiLM; built by data/mimic/build_mimic4.py
out_dir = 'Delphi-mimic4-film'
eval_interval = 500
eval_iters = 50
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi-mimic4'
wandb_run_name = 'mimic4-film-' + str(time.time())

dataset = 'mimic4_labs'
# Effective batch 128 as two micro-batches of 64: at block_size 256 the attention maps of a
# single batch of 128 need ~18 GB of GPU memory, more than a T4 has (~9 GB at 64)
batch_size = 64
gradient_accumulation_steps = 2
block_size = 256  # hospital records are longer than UK Biobank's; 3.5% of patients exceed 256
data_fraction = 1.0

# Same architecture as the synthetic runs
n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.0
weight_decay = 2e-1
vocab_size = 1287  # Delphi's 1270 labels + 17 valued tokens (15 labs, BMI, systolic BP)

# ~98k training patients: 10k iterations of 128 is about 13 passes over the data
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

# FiLM defaults; scripts/run_film_variants.sh overrides these per variant.
# Options are documented on DelphiConfig in model.py.
use_film = True
film_location = 'both'
film_layers = 'all'
film_mode = 'both'
film_context = 'hidden'
film_hidden_dim = 0
film_scale = 0.1
