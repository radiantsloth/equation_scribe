"""
Train a VisionEncoderDecoderModel (ViT encoder + BART decoder) for Image -> LaTeX.

Usage:
    python -m equation_scribe.recognition.train_vision_encoder_decoder \
        --train-jsonl path/to/train_pairs.jsonl \
        --val-jsonl path/to/val_pairs.jsonl \
        --output_dir runs/recog_vit_bart \
        --epochs 5 --batch_size 8 --device cuda

Input JSONL format (one JSON per line):
    {"image": "path/to/crop1.png", "text": "\\nabla\\cdot \\mathbf{E} = \\rho/\\varepsilon_0"}
    {"image": "path/to/crop2.png", "text": "E=mc^2"}
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    AutoConfig,
    AutoFeatureExtractor,
)

# Use the CharTokenizer implemented above or import it if present
try:
    from .tokenizer import CharTokenizer
except Exception:
    # fallback if running as script outside package
    from equation_scribe.recognition.tokenizer import CharTokenizer

# If you want to use your preprocess functions (deskew, normalize), import them:
try:
    from .preprocess import deskew_crop
except Exception:
    # fallback import path
    from equation_scribe.recognition.preprocess import deskew_crop


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic programming Levenshtein distance"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[lb]


class RecognitionJsonlDataset(Dataset):
    """
    Dataset expecting JSONL with {"image": path, "text": latex_text}
    Returns dict with PIL.Image and str text.
    """
    def __init__(self, jsonl_path: str, shuffle: bool = False, deskew: bool = False):
        self.jsonl_path = str(jsonl_path)
        self.entries = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.entries.append(obj)
        if shuffle:
            random.shuffle(self.entries)
        self.deskew = deskew

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        rec = self.entries[idx]
        img_path = rec["image"]
        text = rec["text"]
        img = Image.open(img_path).convert("RGB")
        if self.deskew:
            try:
                img, _ = deskew_crop(img, return_angle=True, expand=True)
            except Exception:
                # if deskew fails, just continue with original
                pass
        return {"image": img, "text": text, "image_path": img_path}


def collate_fn(batch: List[Dict], feature_extractor, tokenizer: CharTokenizer, max_target_length: int = 256):
    """
    Collate:
    - Run feature_extractor on list of images -> pixel_values (torch tensor)
    - Tokenize texts with CharTokenizer -> padded label tensor with -100 for pad (as HF expects)
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    # feature extractor returns pixel_values as a list when return_tensors='pt' is used on a list
    encoding = feature_extractor(images=images, return_tensors="pt")
    pixel_values = encoding["pixel_values"]  # shape (B, C, H, W)

    # Tokenize
    token_ids = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
    # pad to max length in this batch or the provided max_target_length
    max_len = min(max(len(x) for x in token_ids), max_target_length)
    labels = []
    for ids in token_ids:
        if len(ids) > max_len:
            ids = ids[:max_len]
            if ids[-1] != tokenizer.eos_token_id:
                ids[-1] = tokenizer.eos_token_id
        pad_len = max_len - len(ids)
        ids_padded = ids + [tokenizer.pad_token_id] * pad_len
        # Replace pad token id with -100 for computing loss with HuggingFace models
        ids_padded = [i if i != tokenizer.pad_token_id else -100 for i in ids_padded]
        labels.append(ids_padded)

    labels = torch.tensor(labels, dtype=torch.long)

    batch_out = {
        "pixel_values": pixel_values,
        "labels": labels,
        "raw_texts": texts,
    }
    return batch_out


def evaluate(model, dataloader, tokenizer: CharTokenizer, device: torch.device, num_examples: int = 20, gen_kwargs: dict = None):
    model.eval()
    if gen_kwargs is None:
        gen_kwargs = {"max_length": 200, "num_beams": 3}
    total = 0
    exact = 0
    sum_norm_edit = 0.0
    examples = []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            texts = batch["raw_texts"]
            outputs = model.generate(pixel_values=pixel_values, **gen_kwargs)
            # outputs: (B, seq_len)
            for i in range(outputs.shape[0]):
                pred_ids = outputs[i].tolist()
                # remove special tokens via decode
                pred_text = tokenizer.decode(pred_ids, strip_special=True)
                gold_text = texts[i]
                total += 1
                if pred_text.strip() == gold_text.strip():
                    exact += 1
                lev = levenshtein(pred_text, gold_text)
                norm = lev / max(1, len(gold_text))
                sum_norm_edit += norm
                if len(examples) < 5:
                    examples.append((gold_text, pred_text, norm))
            if total >= num_examples:
                break
    avg_exact = exact / total if total > 0 else 0.0
    avg_norm_edit = sum_norm_edit / total if total > 0 else 0.0
    return {"exact_match": avg_exact, "norm_edit": avg_norm_edit, "examples": examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, help="JSONL of train pairs (image,text)")
    parser.add_argument("--val-jsonl", required=True, help="JSONL of val pairs")
    parser.add_argument("--output_dir", default="runs/recog_vit_bart", help="Output dir")
    parser.add_argument("--encoder_model", default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--decoder_model", default="facebook/bart-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--deskew", action="store_true", help="Deskew crops at dataset load time")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # tokenizer
    tokenizer = CharTokenizer()
    # Optionally save tokenizer to output dir
    tokenizer.save(os.path.join(args.output_dir, "char_tokenizer.json"))
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)

    # Feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.encoder_model)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_model, args.decoder_model
    )

    # Resize decoder token embeddings to our tokenizer size
    # Note: this will append randomly initialized embeddings for new tokens.
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = tokenizer.vocab_size

    # Resize token embeddings for decoder to match tokenizer
    # some models put token embeddings on model.model.decoder (HF internals vary)
    try:
        model.decoder.resize_token_embeddings(tokenizer.vocab_size)
    except Exception:
        # fallback: try resize on the full model
        try:
            model.resize_token_embeddings(tokenizer.vocab_size)
        except Exception:
            logger.warning("Resize token embeddings failed; model may not be compatible. Proceeding anyway.")

    model.to(device)

    # Dataset and dataloaders
    train_ds = RecognitionJsonlDataset(args.train_jsonl, shuffle=True, deskew=args.deskew)
    val_ds = RecognitionJsonlDataset(args.val_jsonl, shuffle=False, deskew=args.deskew)

    def collate(batch):
        return collate_fn(batch, feature_extractor, tokenizer, max_target_length=args.max_target_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, collate_fn=collate, num_workers=2)

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    best_val = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if batch_idx % 20 == 0:
                avg = running_loss / 20
                logger.info(f"Epoch {epoch} batch {batch_idx} loss {avg:.4f}")
                running_loss = 0.0

        # evaluation
        model.eval()
        metrics = evaluate(model, val_loader, tokenizer, device, num_examples=200, gen_kwargs={"max_length": 200, "num_beams": 3})
        logger.info(f"Epoch {epoch} eval: exact_match={metrics['exact_match']:.4f}, norm_edit={metrics['norm_edit']:.4f}")
        for g, p, n in metrics["examples"]:
            logger.info("EXAMPLE gold: %s", g)
            logger.info("EXAMPLE pred: %s", p)
            logger.info("norm_edit: %.4f", n)

        # save
        if args.save_every and (epoch % args.save_every == 0):
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            # save tokenizer too
            tokenizer.save(os.path.join(ckpt_dir, "char_tokenizer.json"))
            logger.info("Saved checkpoint to %s", ckpt_dir)

        # track best val by normalized edit
        score = metrics["norm_edit"]
        if best_val is None or score < best_val:
            best_val = score
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save(os.path.join(best_dir, "char_tokenizer.json"))
            logger.info("Saved best model to %s", best_dir)

    logger.info("Training complete. Best val norm_edit=%s", best_val)


if __name__ == "__main__":
    main()
