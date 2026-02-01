import os, json, numpy as np, sentencepiece as spm
from tqdm import tqdm

inp = os.path.expanduser("~/LLM_Train/prepare_data/pretrain_corpus.txt")
spm_model = os.path.expanduser("~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm/tokenizer.model")

out_dir = os.path.expanduser("~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm/")
os.makedirs(out_dir, exist_ok=True)

out_train = os.path.join(out_dir, "train.bin")
out_val   = os.path.join(out_dir, "val.bin")
out_meta  = os.path.join(out_dir, "meta.json")

VAL_FRACTION = 0.01
DTYPE = np.uint16

sp = spm.SentencePieceProcessor(model_file=spm_model)
vocab_size = sp.get_piece_size()

file_size = os.path.getsize(inp)
split_at = int(file_size * (1.0 - VAL_FRACTION))

for p in (out_train, out_val):
    if os.path.exists(p):
        os.remove(p)

train_tokens = 0
val_tokens = 0

with open(inp, "rb") as f, open(out_train, "ab") as ftr, open(out_val, "ab") as fva:
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing(SPM)")
    for line in f:
        pbar.update(len(line))
        s = line.decode("utf-8", errors="ignore").strip()
        if not s:
            continue
        ids = sp.encode(s, out_type=int)
        ids.append(sp.eos_id())  # important separator

        arr = np.asarray(ids, dtype=DTYPE)
        if f.tell() < split_at:
            ftr.write(arr.tobytes()); train_tokens += arr.size
        else:
            fva.write(arr.tobytes()); val_tokens += arr.size
    pbar.close()

meta = {
    "vocab_size": int(vocab_size),
    "dtype": "uint16",
    "val_fraction": VAL_FRACTION,
    "train_tokens": int(train_tokens),
    "val_tokens": int(val_tokens),
    "tokenizer_model": spm_model,
    "bos_id": sp.bos_id(),
    "eos_id": sp.eos_id(),
    "unk_id": sp.unk_id(),
    "pad_id": sp.pad_id(),
}
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)

print("Done.")
print("train tokens:", train_tokens)
print("val tokens:", val_tokens)
print("vocab_size:", vocab_size)
print("Wrote:", out_dir)