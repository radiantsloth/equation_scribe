# Im2LaTeX / Recognizer training guide

This document describes a reproducible path to train an image->LaTeX recognizer (Im2LaTeX-like).

## Recommended approach

1. **Pretrain** on synthetic & public datasets (Im2LaTeX-100k).
2. **Fine-tune** on in-domain crops (synthetic + arXiv/extracted pairs + human corrections).
3. **Postprocess** outputs with a normalization step and SymPy check.

## Data layout

- `detector/data/recognition/images/` — cropped images (png)
- `detector/data/recognition/pairs.jsonl` — JSONL with objects: `{"image":"path","latex":"..."}`

## Model choices

- **Im2LaTeX**: ResNet encoder + Transformer decoder. Good baseline.
- **Pix2Struct / Donut**: end-to-end multimodal transformer variants (may be heavier).
- Use beam search at inference and store top-k outputs.

## Training recipe (Im2LaTeX-style)

- Tokenization: Build a token vocabulary over LaTeX tokens. Keep special tokens for start/end/pad.
- Image preprocessing: resize keeping aspect ratio, pad to fixed height, or use adaptive encoder.
- Loss: cross-entropy with teacher forcing, label smoothing can help.
- Batch size: depends on GPU. Start small (batch=32 or 16).
- Learning rates: use Adam with lr 1e-4 for encoder+decoder fine-tuning; consider warmup.
- Checkpoint & early stopping based on validation sequence accuracy / edit distance.

## Quick start with existing repo (example)

You can use public Im2LaTeX implementations (search GitHub `im2latex-pytorch` or `pysim2latex`) or build a small PyTorch model (ResNet + Transformer).

Example hyperparameters for small experiment:
- epochs: 20 (pretrain on synthetic)
- img size: 256x1024 (height x width), or variable width with bucketing
- optimizer: AdamW, lr=1e-4
- scheduler: linear warmup 5000 steps then cosine decay

## Evaluation

- **Exact match**: percentage of predicted LaTeX exactly equal to gold.
- **Edit distance**: normalized Levenshtein distance.
- **Semantic**: parse both with SymPy (antlr runtime) and compare symbolic equivalence; fallback to numeric random substitution if needed.

## Postprocessing

- Normalize LaTeX tokens: unify `\\left( ... \\right)` variants, remove redundant braces.
- If parse fails, try small local repairs (insert braces, add `\\,` etc.) or ask an LLM to propose small fixes then validate with SymPy.

## Production tips

- Keep top-K hypotheses (beam) and show them in UI; usually the correct answer is within top-3.
- Use mixed batching, augmentations like blur/resize to increase robustness to scan artifacts.
- Fine-tune periodically with human corrections (active learning).

References:
- Im2LaTeX (Harvard / DeepMind style papers)
- Pix2Struct & Donut (Google / NAVER)
