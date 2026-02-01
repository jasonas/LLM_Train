dataset = "soulbotalpha_spm"
model_type = "llama"

block_size = 1024
vocab_size = 32000

n_layer = 24
n_embd = 1024
n_head = 16
n_kv_head = 2
ffn_hidden_size = 6144 # Pushes you toward ~0.5B params
rms_norm_eps = 1e-6
rope_theta = 10000.0
dropout = 0.0
bias = False
checkpointing = True

# Keep tokens/step 1*64*1024 = 65536
batch_size = 1
gradient_accumulation_steps = 64

max_iters = 20000
learning_rate = 2e-5
warmup_iters = 500 # 500 from scratch or even 0 if resuming
lr_decay_iters = max_iters
min_lr = 2e-6
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5

eval_interval = 1000
eval_iters = 50
log_interval = 10

device = "mps"
dtype = "bfloat16" # dtype = "float16" / try bfloat16 instead on MPS (often more stable than float16)
compile = False

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
out_dir = "out_soulbotalpha_spm_llama_05b_r1"