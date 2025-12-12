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
)
from torch.optim import AdamW

# Prefer AutoImageProcessor (replacement for AutoFeatureExtractor)
try:
    from transformers import AutoImageProcessor as _AutoImageProcessor
except Exception:
    # fallback to AutoFeatureExtractor on very old transformers versions
    from transformers import AutoFeatureExtractor as _AutoImageProcessor

class Collator:
    """
    Top-level callable collator so it can be pickled on Windows.
    Stores references to image_processor and tokenizer.
    """
    def __init__(self, image_processor, tokenizer, max_target_length: int = 256):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]

        encoding = self.image_processor(images=images, return_tensors="pt")
        pixel_values = encoding["pixel_values"]

        # Tokenize / pad (robust: supports PreTrainedTokenizerFast or CharTokenizer)
        token_ids = []
        for t in texts:
            # Prefer HF tokenizer API: encode(..., add_special_tokens=True)
            ids = None
            if hasattr(self.tokenizer, "encode"):
                try:
                    ids = self.tokenizer.encode(t, add_special_tokens=True)
                except TypeError:
                    # older/custom char tokenizer signature: add_bos/add_eos
                    ids = self.tokenizer.encode(t, add_bos=True, add_eos=True)
            else:
                # tokenizer callable fallback
                enc = self.tokenizer(t, add_special_tokens=True)
                ids = enc["input_ids"]

            # Ensure ids is a plain Python list
            if isinstance(ids, (np.ndarray,)):
                ids = ids.tolist()
            elif isinstance(ids, torch.Tensor):
                ids = ids.tolist()

            # Make sure BOS/EOS are present if tokenizer defines them
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                if len(ids) == 0 or ids[0] != self.tokenizer.bos_token_id:
                    ids = [self.tokenizer.bos_token_id] + ids
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                if ids[-1] != self.tokenizer.eos_token_id:
                    ids = ids + [self.tokenizer.eos_token_id]

            token_ids.append(list(ids))

        max_len = min(max(len(x) for x in token_ids), self.max_target_length)
        labels = []
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        for ids in token_ids:
            ids = list(ids)
            if len(ids) > max_len:
                ids = ids[:max_len]
                # ensure last token is eos if available
                if eos_id is not None and ids[-1] != eos_id:
                    ids[-1] = eos_id
            pad_len = max_len - len(ids)
            ids_padded = ids + [pad_id] * pad_len
            # replace pad token id with -100 for HF loss
            ids_padded = [i if i != pad_id else -100 for i in ids_padded]
            labels.append(ids_padded)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "raw_texts": texts,
        }

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

# --- helpers: levenshtein and postprocess ---
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance (classic DP)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # ensure la <= lb to use less memory
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,      # deletion
                         cur[j - 1] + 1,   # insertion
                         prev[j - 1] + cost)  # substitution
        prev = cur
    return prev[lb]

import re
def postprocess_latex(s: str) -> str:
    """Simple cleaning of predicted/gold latex strings for fair comparison."""
    if s is None:
        return ""
    s = s.strip()
    # common cleanups: collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # trim stray special tokens often included in HF outputs
    s = s.replace("<s>", "").replace("</s>", "").strip()
    # balance braces roughly (append closing braces)
    opens = s.count("{")
    closes = s.count("}")
    if closes < opens:
        s = s + ("}" * (opens - closes))
    return s

# --- robust evaluate function ---
def evaluate(model, val_loader, tokenizer, device, num_examples: int = 200, gen_kwargs: dict = None):
    """
    Evaluate model on validation loader.
    Returns: dict(metrics) and list of example triples (gold, pred, norm_edit)
    """
    if gen_kwargs is None:
        gen_kwargs = {"max_length": 200, "num_beams": 3, "early_stopping": True}

    model.eval()
    total = 0
    exact_matches = 0
    total_norm = 0.0
    examples = []

    with torch.no_grad():
        for batch in val_loader:
            # Expect batch to contain 'pixel_values' and optionally 'labels'
            pixel_values = batch.get("pixel_values")
            labels = batch.get("labels", None)

            if pixel_values is None:
                # skip malformed batch
                continue

            pixel_values = pixel_values.to(device)

            # generate predictions
            try:
                generated = model.generate(pixel_values=pixel_values, **gen_kwargs)
            except Exception as e:
                # generation failed for this batch; skip
                print("Warning: generation failed on a batch:", e)
                continue

            # generated: tensor (batch_size, seq_len)
            # decode preds robustly
            preds = []
            for gen_ids in generated:
                try:
                    pred_text = tokenizer.decode(gen_ids.cpu().numpy().tolist(), skip_special_tokens=True)
                except TypeError:
                    # older/custom tokenizer
                    try:
                        pred_text = tokenizer.decode(gen_ids.cpu().numpy().tolist(), strip_special=True)
                    except Exception:
                        # last resort: join token ids
                        pred_text = " ".join(map(str, gen_ids.cpu().numpy().tolist()))
                pred_text = postprocess_latex(pred_text)
                preds.append(pred_text)

            # decode gold texts from labels (handle -100)
            golds = []
            if labels is not None:
                # labels shape: (batch, L)
                for lab in labels:
                    lab_ids = lab.cpu().numpy().tolist()
                    # remove mask tokens (-100)
                    lab_ids = [int(x) for x in lab_ids if int(x) != -100]
                    if len(lab_ids) == 0:
                        gold_text = ""
                    else:
                        # decode robustly
                        try:
                            gold_text = tokenizer.decode(lab_ids, skip_special_tokens=True)
                        except TypeError:
                            try:
                                gold_text = tokenizer.decode(lab_ids, strip_special=True)
                            except Exception:
                                gold_text = ""
                    gold_text = postprocess_latex(gold_text)
                    golds.append(gold_text)
            else:
                # no labels present; use empty golds
                golds = [""] * len(preds)

            # Now compare pairwise
            for pred_text, gold_text in zip(preds, golds):
                # ensure both defined
                if gold_text is None:
                    gold_text = ""
                if pred_text is None:
                    pred_text = ""

                total += 1
                if pred_text == gold_text:
                    exact_matches += 1

                # compute normalized edit distance
                dist = levenshtein(pred_text, gold_text)
                denom = max(1, max(len(pred_text), len(gold_text)))
                norm = dist / denom
                total_norm += norm

                # save some examples
                if len(examples) < num_examples:
                    examples.append((gold_text, pred_text, norm))

    # avoid division by zero
    if total == 0:
        return {"exact_match": 0.0, "norm_edit": float("inf"), "examples": examples}

    metrics = {
        "exact_match": exact_matches / total,
        "norm_edit": total_norm / total,
        "examples": examples,
    }
    return metrics


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
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2, help="Number of DataLoader workers")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0,
                    help="Number of initial epochs to keep the encoder frozen. 0 = no freezing.")
    parser.add_argument("--encoder-unfreeze-lr", type=float, default=None,
                    help="LR to use for encoder when unfrozen. If not set, uses args.lr * 0.1.")
    parser.add_argument("--tokenizer-file", default=None, help="Path to trained tokenizer json (e.g., detector/tokenizer/latex_tokenizer.json)")

    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # tokenizer
    tokenizer = None
    if getattr(args, "tokenizer_file", None):
        tk_path = Path(args.tokenizer_file)
        if not tk_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tk_path}")
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tk_path),
                bos_token="<s>",
                eos_token="</s>",
                pad_token="[PAD]",
                unk_token="[UNK]",
            )
            logger.info("Loaded PreTrainedTokenizerFast from %s (vocab_size=%d)", tk_path, tokenizer.vocab_size)
        except Exception as e:
            logger.warning("Failed to load PreTrainedTokenizerFast from %s: %s", tk_path, e)
            tokenizer = None

    # Fallback: if tokenizer is still None, attempt to use existing CharTokenizer in the repo
    if tokenizer is None:
        try:
            # If you have a CharTokenizer implemented (previous pipeline), import it
            from equation_scribe.recognition.tokenizer import CharTokenizer
            tokenizer = CharTokenizer()
            logger.info("Using CharTokenizer fallback (vocab size = %d)", tokenizer.vocab_size)
        except Exception:
            raise RuntimeError("No tokenizer available. Please provide --tokenizer-file pointing to detector/tokenizer/latex_tokenizer.json")


    # Image processor / feature extractor and model
    # Use AutoImageProcessor (new API). For backward compatibility we alias to
    try:
        image_processor = _AutoImageProcessor.from_pretrained(args.encoder_model,use_fast=True)
    except Exception:
        image_processor = _AutoImageProcessor.from_pretrained(args.encoder_model,use_fast=False)
    
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

    # Optionally freeze encoder for the first N epochs
    if args.freeze_encoder_epochs > 0:
        logger.info("Freezing encoder for first %d epoch(s)", args.freeze_encoder_epochs)
        for p in model.encoder.parameters():
            p.requires_grad = False


    # create collator (pickleable top-level class)
    collator = Collator(image_processor, tokenizer, max_target_length=args.max_target_length)

    # Create datasets first (so variables exist)
    train_ds = RecognitionJsonlDataset(args.train_jsonl, shuffle=True, deskew=args.deskew)
    val_ds = RecognitionJsonlDataset(args.val_jsonl, shuffle=False, deskew=args.deskew)

    # Create dataloaders using the collator
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps,
    )

    best_val = None
    for epoch in range(1, args.epochs + 1):
        # If we asked to freeze for N epochs, unfreeze encoder at epoch N+1
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            logger.info("Unfreezing encoder at epoch %d", epoch)
            for p in model.encoder.parameters():
                p.requires_grad = True
            # Note: we keep the same optimizer & LR for simplicity.
            # For most small experiments, this is perfectly fine.

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
        metrics = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            num_examples=200,
            gen_kwargs={"max_length": 200, "num_beams": 3},
        )
        logger.info(
            f"Epoch {epoch} eval: exact_match={metrics['exact_match']:.4f}, "
            f"norm_edit={metrics['norm_edit']:.4f}"
        )
        for g, p, n in metrics["examples"]:
            logger.info("EXAMPLE gold: %s", g)
            logger.info("EXAMPLE pred: %s", p)
            logger.info("norm_edit: %.4f", n)

        # save checkpoint
        if args.save_every and (epoch % args.save_every == 0):
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            # legacy CharTokenizer save (if available)
            try:
                if hasattr(tokenizer, "save"):
                    tokenizer.save(os.path.join(ckpt_dir, "char_tokenizer.json"))
            except Exception:
                pass
            # HF-style save_pretrained (if available)
            try:
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            except Exception:
                pass            
            try:
                image_processor.save_pretrained(ckpt_dir)
            except Exception:
                pass
            logger.info("Saved checkpoint to %s", ckpt_dir)

        # track best val by normalized edit distance (lower is better)
        score = metrics["norm_edit"]
        if best_val is None or score < best_val:
            best_val = score
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            # legacy CharTokenizer save (if available)
            try:
                if hasattr(tokenizer, "save"):
                    tokenizer.save(os.path.join(best_dir, "char_tokenizer.json"))
            except Exception:
                pass
            # HF-style save_pretrained (if available)
            try:
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(os.path.join(best_dir, "tokenizer"))
            except Exception:
                pass            
            try:
                image_processor.save_pretrained(best_dir)
            except Exception:
                pass
            logger.info("Saved best model to %s", best_dir)

    logger.info("Training complete. Best val norm_edit=%s", best_val)

if __name__ == "__main__":
    main()
