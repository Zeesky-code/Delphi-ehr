import os
import time
import math
import json
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
seed = 42

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'delphi'
wandb_run_name = 'run' + str(time.time())
wandb_group = ''  # runs in the same group (e.g. one FiLM variant over several seeds) are averaged in the UI

# data
dataset = 'ukb_simulated_data'
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24

# model
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
vocab_size = 256

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 10000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32'  # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# delphi training
token_dropout = 0.0
t_min = 0.0  # 365.25/12.
mask_ties = True
use_film = False  # FiLM options are documented on DelphiConfig in model.py
film_location = 'both'
film_layers = 'all'
film_mode = 'both'
film_context = 'hidden'
film_hidden_dim = 0
film_scale = 0.1
ignore_tokens = [0]
data_fraction = 1.0
no_event_token_rate = 5


# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
with open('configurator.py') as f:
    exec(f.read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)

# Load numeric values if available (parallel float32 arrays)
train_values_path = os.path.join(data_dir, 'train_values.bin')
val_values_path = os.path.join(data_dir, 'val_values.bin')
if use_film and os.path.exists(train_values_path):
    train_values = np.memmap(train_values_path, dtype=np.float32, mode='r')
    val_values = np.memmap(val_values_path, dtype=np.float32, mode='r')
    # Compute per-token-type statistics for z-score normalization
    values_stats = {}
    has_value = train_values != 0
    valued_tokens = train_data[has_value, 2]
    valued_values = train_values[has_value]
    for tok_id in np.unique(valued_tokens):
        vals = valued_values[valued_tokens == tok_id]
        if len(vals) > 100:  # need enough samples for stable stats
            values_stats[int(tok_id)] = (float(vals.mean()), float(vals.std()))
    print(f"Loaded numeric values for {len(values_stats)} token types")
else:
    train_values = None
    val_values = None
    values_stats = None
    if use_film:
        print("WARNING: use_film=True but no values files found, FiLM will act as identity")

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)

# downsample the data to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


print(f"found vocab_size = {vocab_size}")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, token_dropout=token_dropout, t_min=t_min,
                  mask_ties=mask_ties, use_film=use_film, film_location=film_location,
                  film_layers=film_layers, film_mode=film_mode, film_context=film_context,
                  film_hidden_dim=film_hidden_dim, film_scale=film_scale,
                  ignore_tokens=ignore_tokens)  # start with model_args from command line
FILM_KEYS = ['use_film', 'film_location', 'film_layers', 'film_mode', 'film_context',
             'film_hidden_dim', 'film_scale']

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'] + FILM_KEYS:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16')) 

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

def fetch_batch(split, **kwargs):
    """Sample a batch from a split. V (numeric values) is None unless FiLM values are loaded."""
    data, p2i, values = {'train': (train_data, train_p2i, train_values),
                         'val': (val_data, val_p2i, val_values)}[split]
    ix = torch.randint(len(p2i), (batch_size,))
    batch = get_batch(ix, data, p2i, block_size=block_size, device=device, select='left',
                      no_event_token_rate=no_event_token_rate,
                      values_data=values, values_stats=values_stats, **kwargs)
    return batch if values is not None else (*batch, None)


# helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 2)
        for k in range(eval_iters):
            X, A, Y, B, V = fetch_batch(split, cut_batch=True)
            with ctx:
                logits, loss, _ = model(X, A, Y, B, numeric_values=V, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    import wandb
    raw_model = unoptimized_model if compile else model
    wandb.init(project=wandb_project, name=wandb_run_name, group=wandb_group or None,
               config={**config, 'n_params': raw_model.get_num_params(),
                       'n_film_params': raw_model.get_num_film_params()})

# training loop
X, A, Y, B, V = fetch_batch('train', padding='random', lifestyle_augmentations=True)
t0 = time.time()
iter_times = []
last_val_losses = None
local_iter_num = 0  # number of iterations in the lifetime of this process

val_loss = None
while True:
    metrics = {"iter": iter_num}
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        last_val_losses = losses['val']
        val_loss = losses['val'].sum().item()
        print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")

        metrics.update({
            "train/agg_loss": losses['train'].sum().item(),
            "val/loss":val_loss,
            "val/loss_ce": losses['val'][0].item(),
            "val/loss_dt": losses['val'][1].item()
        })

        if always_save_checkpoint or best_val_loss > val_loss:
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if best_val_loss > val_loss:
            best_val_loss = val_loss

        if iter_num % 10_000 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, att = model(X, A, Y, B, numeric_values=V)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, A, Y, B, V = fetch_batch('train', padding='random', lifestyle_augmentations=True,
                                    cut_batch=True)

        # backward pass, with gradient scaling if training in fp16
        loss = loss['loss_ce'] + loss['loss_dt']
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter_num > 0:  # the first iteration includes warm-up overhead
        iter_times.append(dt)
    if iter_num % log_interval == 0:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        metrics.update({
            "train/loss": lossf,
            "lr": lr,
        })

    if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
        wandb.log(metrics)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# one-line record per run, for comparing variants (see scripts/compare_runs.py)
raw_model = unoptimized_model if compile else model
summary = {
    'out_dir': out_dir,
    'seed': seed,
    'iters': iter_num - 1,
    'best_val_loss': best_val_loss if best_val_loss < 1e9 else None,
    'final_val_loss_ce': last_val_losses[0].item() if last_val_losses is not None else None,
    'final_val_loss_dt': last_val_losses[1].item() if last_val_losses is not None else None,
    'n_params': raw_model.get_num_params(),
    'n_film_params': raw_model.get_num_film_params(),
    'ms_per_iter': 1000 * float(np.median(iter_times)) if iter_times else None,
    **{k: model_args[k] for k in FILM_KEYS},
}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"wrote {os.path.join(out_dir, 'summary.json')}")
if wandb_log:
    wandb.run.summary.update({k: v for k, v in summary.items() if v is not None})
    wandb.finish()
