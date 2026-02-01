import os
import pickle

def main() -> None:
    d = os.path.expanduser("~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm")

    meta = {
        "vocab_size": 32000,
        "dtype": "uint16",
        "tokenizer_type": "sentencepiece",
        "tokenizer_model": os.path.expanduser(
            "~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm/tokenizer.model"
        ),
        "block_size": 1024,
    }

    os.makedirs(d, exist_ok=True)

    out_path = os.path.join(d, "meta.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(meta, f)

    print("Wrote", out_path)

if __name__ == "__main__":
    main()
