import os
import json
import shutil
import argparse
import torch
from safetensors.torch import save_file

import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM

def map_state_dict(sd, n_layer):
    out = {}

    # Embeddings (Doesn't store lm_head.weight to avoid safetensors shared-memory issue)
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]

    # Final norm
    out["model.norm.weight"] = sd["norm.weight"]

    # Export lm_head.weight
    out["lm_head.weight"] = sd["lm_head.weight"].clone()   # <-- add clone

    for i in range(n_layer):
        # Norms
        out[f"model.layers.{i}.input_layernorm.weight"] = sd[f"blocks.{i}.attn_norm.weight"]
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"blocks.{i}.ffn_norm.weight"]

        # Attention projections
        out[f"model.layers.{i}.self_attn.q_proj.weight"] = sd[f"blocks.{i}.attn.wq.weight"]
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = sd[f"blocks.{i}.attn.wk.weight"]
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = sd[f"blocks.{i}.attn.wv.weight"]
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = sd[f"blocks.{i}.attn.wo.weight"]

        # MLP (SwiGLU): gate_proj, up_proj, down_proj
        out[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"blocks.{i}.mlp.w1.weight"]
        out[f"model.layers.{i}.mlp.up_proj.weight"]   = sd[f"blocks.{i}.mlp.w3.weight"]
        out[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"blocks.{i}.mlp.w2.weight"]

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to ckpt.pt")
    ap.add_argument("--spm_model", required=True, help="Path to SentencePiece tokenizer model (tokenizer.model or spm.model)")
    ap.add_argument("--out_dir", required=True, help="Output HF repo folder")
    ap.add_argument("--max_pos", type=int, default=1024)
    args = ap.parse_args()

    ckpt_path = os.path.expanduser(args.ckpt)
    spm_path = os.path.expanduser(args.spm_model)
    out_dir = os.path.expanduser(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ma = ckpt["model_args"]
    sd = ckpt["model"]

    n_layer = int(ma["n_layer"])
    n_head  = int(ma["n_head"])
    n_kv    = int(ma.get("n_kv_head", ma.get("n_kv_heads", 1)))
    dim     = int(ma["n_embd"])
    ffn     = int(ma.get("ffn_hidden_size", ma.get("intermediate_size", 4096)))
    vocab   = int(ma["vocab_size"])
    eps     = float(ma.get("rms_norm_eps", 1e-6))
    theta   = float(ma.get("rope_theta", 10000.0))

    # Build HF config
    hf_cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=dim,
        intermediate_size=ffn,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv,
        rms_norm_eps=eps,
        rope_theta=theta,
        max_position_embeddings=args.max_pos,
        #tie_word_embeddings=True,  # IMPORTANT: ties lm_head to embed_tokens
        tie_word_embeddings=False, # IMPORTANT: now have a real lm_head.weight
        use_cache=True,
        attention_bias=False,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        unk_token_id=1,
    )

    # Instantiate HF model for a shape sanity check (optional)
    hf_model = LlamaForCausalLM(hf_cfg)

    mapped = map_state_dict(sd, n_layer)

    missing, unexpected = hf_model.load_state_dict(mapped, strict=False)
    if unexpected:
        print("Unexpected keys:", unexpected)

    # It's normal that lm_head.weight is missing because we tie embeddings
    # But we DO want most weights to be present.
    # missing_weights = [k for k in missing if k.endswith(".weight") and k != "lm_head.weight"]

    # Simplified 
    missing_weights = [k for k in missing if k.endswith(".weight")]

    if missing_weights:
        print("Missing weights (check these):", missing_weights)

    # Makes the HF repo immediately detectable by llama.cpp
    hf_cfg.architectures = ["LlamaForCausalLM"]

    # Save weights as safetensors (no shared-storage tensors now)
    save_file(mapped, os.path.join(out_dir, "model.safetensors"))

    # Save config.json
    hf_cfg.to_json_file(os.path.join(out_dir, "config.json"))

    # Tokenizer: llama.cpp / HF expects tokenizer.model for LLaMA
    shutil.copyfile(spm_path, os.path.join(out_dir, "tokenizer.model"))

    # Basic tokenizer config
    tok_cfg = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": args.max_pos,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2)

    special_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_map, f, indent=2)

    # Generation defaults (optional but nice)
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    gen_cfg = {"bos_token_id": int(sp.bos_id()), "eos_token_id": int(sp.eos_id())}
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)

    print("Wrote HF model repo to:", out_dir)
    print("Saved:", os.path.join(out_dir, "model.safetensors"))
    print("Saved:", os.path.join(out_dir, "config.json"))
    print("Saved:", os.path.join(out_dir, "tokenizer.model"))

if __name__ == "__main__":
    main()
