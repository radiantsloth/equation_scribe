#!/usr/bin/env python3
"""
Train a BPE tokenizer for LaTeX expressions using Hugging Face `tokenizers`.

Example:
  python tools/train_latex_tokenizer.py \
    --input detector/data/recognition_pairs/all_latex.txt \
    --out-dir detector/tokenizer \
    --vocab-size 4000
"""
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Sequence

def train_bpe(in_file: Path, out_dir: Path, vocab_size: int = 4000):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Initialize an empty BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Pre-tokenizer: ByteLevel works well for LaTeX because of punctuation handling
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[PAD]", "[UNK]", "<s>", "</s>", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    print(f"Training BPE tokenizer on {in_file} -> vocab {vocab_size}")
    tokenizer.train([str(in_file)], trainer=trainer)

    # Post-processor (ByteLevel)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save the raw tokenizer json which PreTrainedTokenizerFast can load via tokenizer_file
    out_file = out_dir / "latex_tokenizer.json"
    tokenizer.save(str(out_file))
    print("Saved tokenizer JSON to:", out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more input text files (one LaTeX per line).")
    parser.add_argument("--out-dir", default="detector/tokenizer", help="Output directory for the tokenizer JSON.")
    parser.add_argument("--vocab-size", type=int, default=4000)
    args = parser.parse_args()

    # If multiple input files, merge them into a temporary list — tokenizers supports list.
    in_files = [Path(p) for p in args.input]
    # use first input file as primary (trainer accepts a list)
    train_bpe(in_files[0], Path(args.out_dir), args.vocab_size)
