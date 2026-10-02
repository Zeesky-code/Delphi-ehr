
import time

out_dir = 'Delphi-labs-baseline'
eval_interval = 250
eval_iters = 25
log_interval = 25

always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi'
wandb_run_name = 'labs-baseline-' + str(time.time())

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
vocab_size = 1280  # 1270 original labels (incl. padding) + 10 lab tokens

learning_rate = 2e-3
max_iters = 3000  # runs overfit after ~2000-2500 iterations on the synthetic data
lr_decay_iters = 3000
min_lr = 2e-4
beta2 = 0.99

warmup_iters = 500

# Ignore padding, sex and lifestyle tokens (same as original) and the lab tokens (model ids
# 1270-1279): labs are inputs only, so the model is never trained to predict or generate them
ignore_tokens = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + list(range(1270, 1280))
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
