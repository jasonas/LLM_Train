import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
    block_size: int = 1024
    vocab_size: int = 50304

    # LLaMA-XS-ish
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 2
    n_embd: int = 1024
    ffn_hidden_size: int = 4096

    # Norms/rope
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0  # Llama2 default is 10000

    # Regularization
    dropout: float = 0.0
    bias: bool = False

    # Optional memory saver
    checkpointing: bool = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    # RoPE cache as cos/sin with shape (seq_len, head_dim/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin

def _apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    # cos/sin: (T, D/2)
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    # broadcast cos/sin: (1,1,T,D/2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).view(B, H, T, D)
    return y

class CausalSelfAttentionGQA(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.kv_repeat = cfg.n_head // cfg.n_kv_head

        # q has n_head, k/v have n_kv_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _ensure_cache(self, T, device, dtype):
        if self.rope_cos is None or self.rope_cos.shape[0] < T or self.rope_cos.dtype != dtype:
            cos, sin = _build_rope_cache(T, self.head_dim, self.cfg.rope_theta, device, dtype)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x):
        B, T, C = x.size()
        device = x.device
        dtype = x.dtype
        self._ensure_cache(T, device, dtype)

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)          # (B, H, T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)       # (B, Hkv, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)       # (B, Hkv, T, D)

        # RoPE on q and k
        q = _apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = _apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])

        # Repeat kv heads to match q heads (GQA)
        if self.kv_repeat != 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)  # (B, H, T, D)
            v = v.repeat_interleave(self.kv_repeat, dim=1)  # (B, H, T, D)

        # Use SDPA if available; it’s typically fastest and handles causal masking
        try:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0.0,
                is_causal=True
            )  # (B, H, T, D)
        except Exception:
            # Manual attention fallback
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
            # build causal mask if needed
            if self.mask is None or self.mask.size(-1) < T:
                self.mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))[None, None, :, :]
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.wo(y))
        return y

class SwiGLU(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.ffn_hidden_size, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embd, cfg.ffn_hidden_size, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.ffn_hidden_size, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = CausalSelfAttentionGQA(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.config = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        # Collect all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate into decay/no_decay
        decay_params = []
        nodecay_params = []
        for pn, p in param_dict.items():
            # No weight decay for 1D params (norm scales) and biases
            if p.dim() < 2 or pn.endswith(".bias"):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )
        return optimizer

    def forward(self, idx, targets=None):
        # idx: (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        x = self.tok_emb(idx)  # (B, T, C)
        x = self.drop(x)

        if self.config.checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
