import os, json, pickle

def main() -> None:
    d = os.path.expanduser("~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm")
    meta_json_path = os.path.join(d, "meta.json")

    with open(meta_json_path, "r", encoding="utf-8") as f:
        mj = json.load(f)

    meta = {
        "vocab_size": int(mj["vocab_size"]),
        "dtype": mj["dtype"],
        "tokenizer_type": "sentencepiece",
        "tokenizer_model": os.path.expanduser(mj["tokenizer_model"]),
        "block_size": 1024,
        "bos_id": int(mj["bos_id"]),
        "eos_id": int(mj["eos_id"]),
        "unk_id": int(mj["unk_id"]),
        "pad_id": int(mj["pad_id"]),
    }

    out_path = os.path.join(d, "meta.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(meta, f)

    print("Wrote", out_path)

if __name__ == "__main__":
    main()
