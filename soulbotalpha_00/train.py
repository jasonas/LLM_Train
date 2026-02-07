import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# From model import GPTConfig as GPT2Config, GPT as GPT2Model
from model_llama import LlamaConfig as LlamaConfig, Llama as LlamaModel

# Default config values 
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # If True, script exits right after the first eval
always_save_checkpoint = True # If True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# Logging wandb 
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Tokenizer / vocab
# NOTE: will normally be overridden by meta.pkl if present
vocab_size = 50304

# Model type: 'llama' or 'gpt2'
model_type = 'llama'

# model defaults
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # For pretraining 0 is good, for finetuning try 0.1+
bias = False # Bias inside LayerNorm and Linear layers

# LLaMA-specific (safe defaults; overridden by config when model_type='llama')
n_kv_head = 2
ffn_hidden_size = 4096
rms_norm_eps = 1e-6
rope_theta = 10000.0
checkpointing = False

# Adamw optimizer
learning_rate = 2e-4 # Max learning rate
max_iters = 600000 # Total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # Clip gradients at this value, or disable if == 0.0

# Learning rate decay settings
decay_lr = True # Whether to decay the learning rate
warmup_iters = 500 # How many steps to warm up for
lr_decay_iters = 600000 # Should be ~= max_iters per Chinchilla
min_lr = 2e-5 # Minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# System
device = 'cuda' # Examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Capture (potential) config keys for logging (must be done before configurator overrides)
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

# Override config from command line or config file
exec(open('configurator.py').read())

# Freeze config for logging
config = {k: globals()[k] for k in config_keys}

# I/O Setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

# Device_type for autocast & optimizer choices
if isinstance(device, str) and device.startswith('cuda'):
    device_type = 'cuda'
elif device == 'mps':
    device_type = 'mps'
else:
    device_type = 'cpu'

# TF32 only makes sense on CUDA
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Autocast context
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # Recreate memmap each batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if meta_vocab_size is not None:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model selection + model args 
model_type = globals().get("model_type", "llama")

if model_type == "llama":
    ModelConfig = LlamaConfig
    Model = LlamaModel
    model_args = dict(
        block_size=block_size,
        vocab_size=vocab_size,  # will be overridden below if meta.pkl exists
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        ffn_hidden_size=ffn_hidden_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        dropout=dropout,
        bias=bias,
        checkpointing=checkpointing,
    )
else:
    ModelConfig = GPT2Config
    Model = GPT2Model
    model_args = dict(
        block_size=block_size,
        vocab_size=vocab_size,  # may be overridden below
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )

# Model init
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    # Choose vocab size
    if meta_vocab_size is None:
        print(f"no meta.pkl found; using vocab_size={vocab_size} from config")
        model_args['vocab_size'] = vocab_size
    else:
        model_args['vocab_size'] = meta_vocab_size

    gptconf = ModelConfig(**model_args)
    model = Model(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # Force key attrs to match checkpoint
    base_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    llama_keys = ['n_kv_head', 'ffn_hidden_size', 'rms_norm_eps', 'rope_theta', 'checkpointing']
    keys = base_keys + (llama_keys if model_type == "llama" else [])
    for k in keys:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]

    gptconf = ModelConfig(**model_args)
    model = Model(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    assert model_type != "llama", "init_from='gpt2*' is only supported when model_type='gpt2'"
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # this is GPT-2 specific
    override_args = dict(dropout=dropout)
    model = GPT2Model.from_pretrained(init_from, override_args)  # may not exist in your fork
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# Crop down the model block size if desired (GPT only; LLaMA implementation may not support)
if hasattr(model, "crop_block_size"):
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size

model.to(device)

# GradScaler: only enabled for CUDA float16
scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile the model (PyTorch 2.x)
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

# Helps estimate loss over either split
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Evaluate in fp32 for a stable, comparable metric
            with nullcontext():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Training Loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

# Report parameter count
if hasattr(raw_model, "get_num_params"):
    nparams = raw_model.get_num_params()
else:
    nparams = sum(p.numel() for p in raw_model.parameters())
print(f"number of parameters: {nparams/1e6:.2f}M")

while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_iter_{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    loss_accum = 0.0

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')

        if not torch.isfinite(loss):
            print(f"Non-finite loss at iter={iter_num}, micro_step={micro_step}: {loss.item()}")
            raise SystemExit(1)

        loss_accum += loss.detach().item()

        if device_type == 'cuda' and dtype == 'float16':
            scaler.scale(loss).backward()
        else:
            loss.backward()
    
    if grad_clip != 0.0:
        if device_type == 'cuda' and dtype == 'float16':
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if device_type == 'cuda' and dtype == 'float16':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    if device_type == 'cuda':
        torch.cuda.synchronize()    # Timing & logging

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5 and hasattr(raw_model, "estimate_mfu"):
            try:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            except Exception:
                pass
        
        lossf = loss_accum  # This is the true mean loss over the whole optimizer step
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
