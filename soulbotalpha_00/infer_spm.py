import os
import json
import argparse
import torch

# SentencePiece (the standard LLaMA tokenizer path)
import sentencepiece as spm

from model_llama import LlamaConfig as GPTConfig, Llama as GPT

def load_spm_tokenizer(meta_json_path: str):
    """
    Expects meta.json created by SPM bin builder to contain either:
      - "spm_model_path": "/abs/or/rel/path/to/spm.model"
    or
      - "tokenizer_model": "/abs/or/rel/path/to/spm.model"
    Falls back to sibling "spm.model" next to meta.json.
    """
    meta_json_path = os.path.expanduser(meta_json_path)
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    spm_path = meta.get("spm_model_path") or meta.get("tokenizer_model")
    if spm_path is None:
        # common fallback: spm.model sits next to meta.json
        spm_path = os.path.join(os.path.dirname(meta_json_path), "spm.model")

    spm_path = os.path.expanduser(spm_path)
    if not os.path.isabs(spm_path):
        spm_path = os.path.join(os.path.dirname(meta_json_path), spm_path)

    if not os.path.exists(spm_path):
        raise FileNotFoundError(
            f"Could not find SentencePiece model. Looked for: {spm_path}\n"
            f"Edit meta.json to include 'spm_model_path' or place 'spm.model' next to meta.json."
        )

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    return sp, spm_path, meta

def encode_prompt(sp: spm.SentencePieceProcessor, prompt: str, add_bos: bool):
    ids = sp.EncodeAsIds(prompt)
    if add_bos:
        bos_id = sp.bos_id()
        # Some SPMs return -1 if BOS is not defined
        if bos_id != -1:
            ids = [bos_id] + ids
    return ids

def decode_ids(sp: spm.SentencePieceProcessor, ids):
    # If BOS/EOS exist, stripping them improves readability
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()
    if bos_id != -1 and len(ids) and ids[0] == bos_id:
        ids = ids[1:]
    # Sometimes generation can include EOS; truncate for clean output
    if eos_id != -1 and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    return sp.DecodeIds(list(ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="out_soulbotalpha_spm_llama_05b_r1/ckpt.pt")
    ap.add_argument(
        "--meta_json",
        type=str,
        default=os.path.expanduser("~/LLM_Test/nanoGPT/data/soulbotalpha_spm/meta.json"),
        help="Path to meta.json produced by the SPM bin builder (contains or sits next to spm.model).",
    )
    ap.add_argument("--prompt", type=str, default="Hello! Write a short paragraph about ")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="mps")  # mps/cpu/cuda
    ap.add_argument("--add_bos", action="store_true", help="Prepend BOS token if tokenizer defines one.")
    ap.add_argument("--eos_stop", action="store_true", help="Stop generation early if EOS is generated.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Load SentencePiece tokenizer
    sp, spm_path, meta = load_spm_tokenizer(args.meta_json)

    # Load checkpoint
    ckpt = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
    model_args = ckpt["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model.to(device)

    # Encode prompt -> ids
    ids = encode_prompt(sp, args.prompt, add_bos=args.add_bos)
    if len(ids) == 0:
        raise ValueError("Prompt encoded to zero tokens. Try a different prompt.")

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)

    eos_id = sp.eos_id()

    # Generate (optionally stop on EOS)
    with torch.no_grad():
        if not args.eos_stop or eos_id == -1:
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )[0].tolist()
        else:
            # Manual loop to allow early stop on EOS while reusing your model.forward
            idx = x
            for _ in range(args.max_new_tokens):
                idx_cond = idx[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / max(args.temperature, 1e-8)

                if args.top_k is not None and args.top_k > 0:
                    v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
                idx = torch.cat((idx, next_id), dim=1)

                if int(next_id.item()) == eos_id:
                    break
            y = idx[0].tolist()

    out_text = decode_ids(sp, y)
    print(out_text)

if __name__ == "__main__":
    main()
