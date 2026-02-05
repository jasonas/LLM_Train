import os, sentencepiece as spm

inp = os.path.expanduser("~/LLM_Train/prepare_data/pretrain_corpus.txt")
out_dir = os.path.expanduser("~/LLM_Train/soulbotalpha_00/data/soulbotalpha_spm")
os.makedirs(out_dir, exist_ok=True)

spm.SentencePieceTrainer.train(
    input=inp,
    model_prefix=os.path.join(out_dir, "tokenizer"),
    vocab_size=32000,
    model_type="bpe",
    character_coverage=0.9995,

    # IDs 
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,

    # IMPORTANT for LLaMA-ish behavior + llama.cpp stability
    byte_fallback=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    add_dummy_prefix=False,

    # Optional but often helpful on big corpora
    shuffle_input_sentence=True,
)

print("Wrote:", os.path.join(out_dir, "tokenizer.model"))
print("Wrote:", os.path.join(out_dir, "tokenizer.vocab"))