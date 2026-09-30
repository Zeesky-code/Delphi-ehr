
import time

out_dir = 'Delphi-film'
eval_interval = 250
eval_iters = 25
log_interval = 25

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi'
wandb_run_name = 'film-' + str(time.time())

# Point to the new lab-extended dataset
dataset = 'ukb_simulated_data_with_labs'
batch_size = 128
block_size = 96
data_fraction = 1.0

# Same architecture as the original demo
n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.0
weight_decay = 2e-1
vocab_size = 1280  # was 1270, +10 lab tokens

learning_rate = 2e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 2e-4
beta2 = 0.99

warmup_iters = 500

# Ignore padding and lifestyle tokens (same as original)
ignore_tokens = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

use_film = True