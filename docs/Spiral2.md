# Spiral 2 — Summary / high level goals

## Primary goal. 
Go from detected equation boxes (from Spiral 1 detector) to robust LaTeX output for each box so that the human-in-the-loop UI can present a good first draft. Secondary goals: build a recognition training pipeline with synthetic + real data, create a symbol glossary and a validator that checks LaTeX for parseability and symbol consistency, and tie these pieces into the GUI.

Success criteria (minimally):

* For printed/scanned equations, the recognition model produces correct LaTeX (or a close LaTeX that can be corrected with small edits) for ≥ ~75% of test equations (on held-out realistic papers).

* The GUI can accept a saved detection box, show the rendered LaTeX, accept edits, validate via SymPy / parser, and save the vetted LaTeX and glossary entry.

## Components to build

## 1 — recognition.preprocess : crop, normalize, deskew and augment equation crops

**Purpose**. Convert detection boxes (pixel coordinates) into canonical recognition images suitable for the image→LaTeX model. Do deskewing, padding, contrast normalization, and optional binarization.

**What to implement (files)**

* equation_scribe/recognition/preprocess.py:

  * crop_page_image(page_img: PIL.Image, bbox: Tuple[int], margin: int=5) -> PIL.Image

  * deskew_crop(img: PIL.Image) -> Tuple[PIL.Image, float]

  * normalize_for_recognition(img: PIL.Image, target_h: int | None=64, preserve_aspect=True) -> np.ndarray (return float32 normalized tensor)

  * augment_for_recognition(img: PIL.Image, args) -> PIL.Image (rotation, blur, noise, contrast)

**Unit tests**

* tests/test_recognition_preprocess.py

  * Given a small synthetic image and known bbox, assert cropped region has expected shape and that deskew_crop returns small angle (zero for non-rotated).

  * Normalization: output dtype, shape constraints, and pixel value ranges.

**Integration test**

* Run crop_page_image -> deskew_crop -> normalize_for_recognition on a real crop from a PDF page and ensure the preprocessed image can be consumed (numpy shape etc).

**Data**

* Use crops from synthetic generator in detector/data/images/tiles_* and also a handful of real cropped boxes from sample PDF.

## 2 — recognition.dataset : recognition dataset & pair generation (image → LaTeX)

**Purpose.** Generate training data: (crop_image, latex_string) pairs. This includes the synthetic generator output and conversion of real annotated papers (Im2LaTeX / CROHME where applicable) into the same format.

**What to implement**

* equation_scribe/recognition/dataset.py:

  * generate_recognition_pairs_from_coco(coco_json, images_dir, out_pairs_jsonl) — for synthetic COCO; crop images and store metadata and label

  * pairs_to_tfrecord or pairs_to_torchdataset (PyTorch Dataset wrappers)

  * small script create_recognition_pairs.py that produces a pairs_train.jsonl and pairs_val.jsonl

**Unit tests**

* Given a minimal COCO with one annotated image, generate_recognition_pairs_from_coco produces expected JSONL with a valid image file and latex entry.

**Datasets**

* Synthetic dataset you already generate (best first).

* Public: Im2LaTeX-100k (common benchmark for image→LaTeX).

* CROHME (handwritten equations) — optional later.

* Kaggle LaTeX datasets / Mathpix exports if available.

* Also collect a small set of real PDFs from your PAPERS_ROOT and hand annotate 50–200 boxes as validation.

**Quality tip**

* Export both the raw crop and a “normalized” recognition image so you can experiment with model inputs.

## 3 — Recognition model baseline: CNN encoder + Transformer decoder

**Purpose**. Implement and train an image→LaTeX baseline. Modern architectures are: convolutional encoder (ResNet-like), flatten + positional encoding, transformer decoder (autoregressive) that outputs LaTeX tokens.

**Options**

* Use an existing open-source im2latex implementation (ResNet+LSTM+attention) as a baseline, or a Transformer-based variant (CNN backbone + Transformer decoder) that’s easier to scale.

* Consider using Pix2Struct / Pix2Seq style if you want to integrate with LLM-style decoders later. But for now, a classical im2latex transformer is robust.

**What to implement**

* equation_scribe/recognition/model.py:

    * RecognitionModel (PyTorch) with:

    * Encoder: small ResNet or pretrained CNN to produce feature map

    * Flatten + linear projection + positional encoding

    * TransformerDecoder (PyTorch nn.TransformerDecoder) or a HuggingFace Bart/T5 style decoder

    * tokenizer.py: LaTeX tokenizer (subword / char-level) — simplest: char-level or token-level with a vocabulary built from training labels; later move to subword.

    * equation_scribe/recognition/trainer.py:

    * Data loader, training loop, checkpointing, eval.

**Unit tests**

* tests/test_recognition_model.py:

* Feed a batch of random images and token targets; assert forward pass works and shapes make sense.

* Small overfit test: train for ~20 steps on 10 pairs and assert loss decreases.

**Integration tests**

Train short tiny model on 100 pairs and run decoding on a small validation set. Save checkpoints.

Evaluation metrics

Exact-match LaTeX string accuracy.

Normalized edit distance (Levenshtein).

BLEU or chrF.

Semantic check: parse both predicted and target via SymPy/latex2sympy (if possible) and compare canonical forms (this will be a later metric).

Dataset notes

Start with synthetic pairs; then fine-tune on Im2LaTeX and a small set of real annotated pairs.

4 — Recognition inference & beam search

Purpose. Implement greedy and beam decoding to produce LaTeX sequences during inference and glue into the GUI API.

What to implement

recognition/infer.py:

decode_greedy and decode_beam (beam size 5).

Postprocessing to remove special tokens and handle common LaTeX normalization (e.g., \ spacing).

Unit tests

Confirm decoder outputs with known small examples.

End-to-end test: preprocessed crop → model → LaTeX string.

Integration

Add API endpoint in backend to call recognition on a crop: POST /recognize returning {latex: ..., score: ...}. The UI will call this when a user selects “Recognize” for a box.

5 — Symbol glossary, consistency, and validator module

Purpose. Ensure each finalized equation has properly defined symbols, and provide a module that scans a paper’s equations and builds a glossary (symbol → definition). Also provide a validator that checks the produced LaTeX for parse errors and attempts to canonicalize equations (via SymPy or LLM).

What to implement

equation_scribe/recognition/validator.py:

parse_latex_to_sympy(latex: str) -> SympyExpr or None (uses sympy.parsing.latex.parse_latex or sympy+antlr if needed).

validate_syntax(latex) -> (ok:bool, errors:list)

symbol_extractor(latex) -> set(symbol_names)

equation_scribe/recognition/glossary.py:

update_glossary(profile_dir, latex, extracted_symbols, definition=None, source='auto'|'human')

resolve_symbol(symbol) to fetch definition from glossary or prompt LLM to propose a definition

Unit tests

Ensure symbol_extractor finds expected tokens from sample LaTeX (e.g., \mathbf{E}, \rho, \varepsilon_0).

Validate validate_syntax correctly flags malformed LaTeX strings.

Integration

When user approves an equation, the backend calls validator:

If syntax parses, add to profile JSONL with validated: True.

If parse fails, present error and suggestions (LLM-based suggestions can be a later spiral).

Notes on LLMs

For semantic validation or suggestions (if SymPy fails), you can use an LLM to propose a corrected LaTeX. That’s a separate module (recognition/llm_helper.py) and can be an optional component.

6 — GUI integration & endpoints

Purpose. Allow recognition and validation to be launched from the front-end. When user selects a box and clicks “Recognize”, the crop is sent to the backend and the returned LaTeX is shown in the LaTeX editor. User can then “Check” to run validator and “Approve & Save” to append to profile.

What to implement

Backend endpoints (FastAPI):

POST /recognize with {paper_id, page_index, bbox} returns {latex, score}.

POST /validate with {latex} returns {ok, errors, sympy_expr}.

POST /save_equation appends record to the profile (existing flow).

Frontend UI updates:

Add a “Recognize” button on the side panel.

When recognition response arrives, render LaTeX in preview pane using KaTeX.

Add validation feedback (green check / red error) and a button “Ask LLM to propose fix” (Optional).

Unit tests

Mock back-end responses to test the front-end behaviors.

Backend unit tests for each endpoint.

Integration test

Manual E2E test: choose a page, click Recognize on a box, validate, approve, and ensure JSONL updated.

7 — Ensemble validation: small LLM ensemble for consistency & symbol checking

Purpose. Use an LLM (or ensemble) to (a) re-check LaTeX for likely misrecognitions, (b) propose symbol definitions, and (c) propose a corrected LaTeX when parser fails.

Options

Use open-source models you can call via local API: e.g., Llama 2 (or other open LLMs) with prompt templates for math. Or use specialized models like Pix2Struct if you want vision+language. Start simple: use a small LLM prompt that checks tokens and suggests likely fixes.

Keep LLM usage optional and slow-path — rely primarily on SymPy for syntactic/semantic checks.

What to implement

equation_scribe/recognition/llm_validate.py:

suggest_fix(latex, context_text=None) -> candidate_latex (calls LLM)

explain_differences(pred, gold) for UI display

Tests

Mock LLM responses. Unit tests ensure the function handles non-answers cleanly.

8 — Training/CI & Evaluation scripts

Purpose. Standardize training runs and evaluation so results are reproducible and tracked.

What to implement

equation_scribe/recognition/evaluate.py:

Computes exact-match, normalized edit distance, BLEU/chrF.

Semantic metric: attempt parse and compare sympy canonical representations.

tools/run_recognition_train.sh or .ps1:

Wrap dataset creation, training, evaluation, save metrics & checkpoints.

Add minimal sample dataset and small CI test that trains a tiny model (smoke test).

Unit tests

tests/test_evaluate.py: compute metrics for a few ground truth/predicted pairs.

9 — Dataset expansion & active learning

Purpose. Move from synthetic-only training to improving the model with real examples and active learning (human corrections).

What to implement

equation_scribe/recognition/active_learning.py:

Periodically sample low confidence predictions and add them to an annotation queue.

When a user corrects LaTeX in the GUI, append that pair to profiles_root and a manual_pairs.jsonl for training.

Data ingestion scripts to convert Im2LaTeX and other public datasets to your pair format.

Tests

Simulate user edits and ensure corrected pair is appended and can be re-used for training.

Suggested iteration order (small spirals)

Each iteration ends with a simple test you can run locally.

Iteration A — Preprocess + pair generation (fast)

Implement preprocess and dataset pair generation.

Unit tests: crop + deskew + normalization.

Minimal output: pairs_train.jsonl and small sanity check.

Why first? Recognition depends on clean inputs. This is fast and enables training.

Iteration B — Small recognition baseline (tiny model)

Implement RecognitionModel and trainer.

Overfit test: train on 20 pairs and ensure loss decreases.

Minimal output: checkpoint + script to decode a few examples.

Why second? You’ll get a working pipeline end-to-end quickly.

Iteration C — Beam search & GUI hook

Add inference code & backend /recognize.

Wire the GUI “Recognize” button to the backend.

Minimal output: UI shows predicted LaTeX and preview render.

Iteration D — Validator & glossary

Implement validator (SymPy parse) and glossary.

UI check/approve flow saving to profile JSONL.

Minimal output: approved LaTeX saved; parser success/failure reported.

Iteration E — Improve dataset & training

Add Im2LaTeX fine-tuning and synthetic augmentation pipeline.

Implement evaluation scripts and run a real validation run.

Minimal output: trained model with meaningful metrics.

Iteration F — LLM assist & active learning

Add LLM suggestion pipeline and active-learning queue.

Collect corrections and retrain.

Datasets & where to get real data

Im2LaTeX-100k — standard dataset of formula images paired with LaTeX. (Great for printed equation recognition baseline.)

CROHME — handwritten math (optional later).

Kaggle LaTeX datasets — check public kaggle for LaTeX formula datasets.

ArXiv / IEEE papers — extract equations from born-digital PDFs (pdfplumber / pymupdf) and collect ground-truth LaTeX where available. (This is the best “real” test set.)

Your synthetic generator (customized to create matrices, fractions, sub/superscripts, integrals — heavy math constructs).

Metrics / evaluation

Exact Match (EM): percent of exact LaTeX string matches.

Normalized Edit Distance: Levenshtein / length normalization.

BLEU / chrF: token-level overlap (helpful but imperfect).

Sympy Semantic Equivalence: parse predicted and gold to SymPy, compare canonical forms (best for algebraic equivalence).

Per-structure accuracy: accuracy on fractions, superscripts, sums, integrals separately (inspect confusion matrix).

For early rounds use EM and edit-distance; add SymPy equivalence when parsing is stable.

Practical advice / tooling & packages

Framework: PyTorch (you already use it). Hugging Face transformers for decoder/seq2seq is convenient.

Tokenizer: Start with character-level tokenizer or byte-pair on LaTeX tokens; char-level is simplest to begin.

Model: Small ResNet + Transformer decoder is a robust baseline. You can later swap encoder with a pretrained ViT/CNN.

Compute: recognition training benefits from GPU, but early small tests are fine on CPU.

Caching: cache rendered LaTeX images when generating synthetic data to avoid rerendering identical expressions.

SymPy/ANTLR: ensure antlr4-python3-runtime is installed for SymPy latex parsing.

Minimal first micro-task (pick this to start)

Implement equation_scribe/recognition/preprocess.py with:

crop_page_image(), deskew_crop(), normalize_for_recognition()
Add unit tests that:

Create a page image with a rotated formula (synthetic), crop the bbox, run deskew_crop, and assert the returned angle is non-zero and the deskewed image is near-upright.

Assert normalized output shape & dtype.

Why? This is fast to implement and gives you a deterministic building block for the recognition model.

File layout suggestions
equation_scribe/
  recognition/
    __init__.py
    preprocess.py
    dataset.py
    tokenizer.py
    model.py
    trainer.py
    infer.py
    validator.py
    glossary.py
    llm_helper.py   # optional
    evaluate.py
tools/
  check_prereqs.py
  run_demo.ps1
detector/    # existing
frontend/    # existing


Quick summary — the plan (priority order)
Improve targets (tokenizer) — move from char-level to a small subword tokenizer (BPE / SentencePiece). This shortens sequences and helps the decoder learn grammar.

Scale & diversify synthetic data — programmatically generate many distinct LaTeX expressions (hundreds → thousands), varying fonts, sizes, DPI, noise, rotations, blur, contrast, deskew. (Not just 12 equations.)

Training recipe changes — lower LR, longer training, keep encoder frozen for a few epochs, use beam search with no_repeat_ngram_size, and monitor validation with cleaned decoding (skip_special_tokens=True).

Improve decoding + postprocessing — beam search, n-gram blocking, postprocess to strip special tokens and fix common syntax errors (balance braces, collapse repeated chars).

Add robust validation — candidate-beam check via LaTeX rendering (pdflatex) or SymPy parse; pick best beam that passes render/parse or lowest edit distance to a grammar-correct form.

Backend inference endpoint + GUI flow — add /recognize endpoint that returns latex and a rendered image (png) for the GUI to show; add a “validate/render” button in the UI.

Persistence for RAG — save approved LaTeX in the paper profile JSONL, plus a symbol glossary and canonical equation ids. (Later: build embeddings / FAISS.)

If you do the above in order, you’ll rapidly get a stable, testable pipeline that the GUI can use for human-in-the-loop verification.

Why these items — one line each
Tokenizer: char tokens are too long; subwords shorten sequences and let decoder model syntax better.

Bigger & diverse data: the current model memorizes; generalization requires variety and scale.

Training recipe: freeze/unfreeze + smaller LR prevents instability when decoder head is randomly initialized.

Decoding & postprocessing: reduces repeated tokens and fixes simple LaTeX glitches.

Validation via rendering/parse: prevents storing broken LaTeX (important for RAG).

Integration: GUI needs a single, reliable recognize call that returns validated LaTeX + rendering.

Concrete code changes and where to apply them
Below are the pieces you can apply right now. I list file names and the minimal, pragmatic code to add. I assume your repo layout is the one you already used (e.g., equation_scribe/recognition/, detector/, backend/).

A. Train a subword tokenizer (fast, reliable option)
Use the tokenizers library (Hugging Face) to train a BPE tokenizer on a corpus of LaTeX strings. You must generate a file all_latex.txt with one LaTeX expression per line — synthetic generation can produce that.

Create tools/train_latex_tokenizer.py (run once):

# tools/train_latex_tokenizer.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from pathlib import Path

def train_bpe(in_files, out_dir="detector/tokenizer", vocab_size=4000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>"
    ])
    tokenizer.train(files=in_files, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(str(out_dir / "latex_tokenizer.json"))
    print("Saved tokenizer to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out-dir", default="detector/tokenizer")
    p.add_argument("--vocab-size", type=int, default=4000)
    args = p.parse_args()
    train_bpe(args.input, args.out_dir, args.vocab_size)
Then load in training code:

from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="detector/tokenizer/latex_tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="[PAD]",
    unk_token="[UNK]",
)
Why this: PreTrainedTokenizerFast wraps the fast tokenizers JSON and is compatible with Transformers decode(..., skip_special_tokens=True).

Vocab size: try 2000–4000 for LaTeX; large enough to absorb common macros.

B. Expand synthetic data (what to change in synthetic_coco.py)
Instead of 12 fixed equations, generate many LaTeX expressions via templates. Keep them programmatic so you can produce thousands.

Examples of templates (sketch):

TEMPLATES = [
    r"E = m c^2",
    r"\frac{d}{dx} \sin x = \cos x",
    r"\int_{0}^{\infty} e^{-x^{2}} dx = \frac{\sqrt{\pi}}{2}",
    r"\sum_{n=1}^{\infty} \frac{1}{n^{2}} = \frac{\pi^{2}}{6}",
    r"\lim_{x\to 0} \frac{\sin x}{x} = 1",
    r"\mathbf{F} = m \mathbf{a}",
    r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
]
# but create variations programmatically:
def sample_fraction(depth=2):
    # recursively create nested fractions and sums/limits
    ...

# generate 3000+ distinct formulas by combining templates with random variables, subscripts,
# exponents, nested fractions, trig, roots, etc.
Key augmentations:

Fonts: Times, Computer Modern, bold/italic.

Sizes: tiny → big.

DPI: 150, 200, 300.

Noise: Gaussian blur, JPEG compression 50–95, salt&pepper, contrast/brightness.

Deskew: small rotations ±0–12° and deskew variant.

Add distractor characters on the page: text snippets or figures so crops are realistic.

Make synthetic_coco produce the recognition_pairs/all.jsonl with fields:

{"image":"crops/crop_0001.png", "latex":"\\frac{d}{dx}\\sin x = \\cos x", "paper_id":"paper001", "page_index":0}
C. Training script changes (train_vision_encoder_decoder.py)
1) Use the new tokenizer. Replace any CharTokenizer usage with PreTrainedTokenizerFast loaded above.

2) Generation config (use beam search + repetition penalty):

Where you call model.generate(...), set:

gen_kwargs = {
    "max_length": args.max_target_length,
    "num_beams": 5,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
    "length_penalty": 0.8,
}
outputs = model.generate(pixel_values=pixel_values, **gen_kwargs)
pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
3) Loss / LR / schedule:

Use args.lr = 1e-5 as default.

Keep freeze encoder for first 2 epochs (you already implemented).

Keep gradient clipping and use AdamW with weight_decay 0.01.

4) Postprocess predictions before eval:

Add a postprocess function:

import re

def postprocess_latex(s):
    # strip special tokens
    s = s.strip()
    # remove repeated spaces
    s = re.sub(r"\s+", " ", s)
    # balance braces (quick heuristic)
    openb = s.count("{")
    closeb = s.count("}")
    if closeb < openb:
        s = s + "}"*(openb-closeb)
    # remove stray leading/trailing symbols that are obviously spurious
    s = s.strip(" <s></s>")
    return s
Call this on both gold and predicted strings before edit distance.

5) Candidate beam verification: During evaluation, for each image, generate top-k (via num_return_sequences=k or loop beams) and postprocess each beam; pick the first beam that passes rendering/parse (next section) or with min edit distance.

D. Validation via pdflatex render (backend helper)
Add a helper recognition/validate_latex.py:

import subprocess, tempfile, os, shlex

TEX_TEMPLATE = r"""\documentclass[preview]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}
%s
\end{document}
"""

def latex_renders_ok(latex):
    # wrap in $...$ if inline macros expected, or keep as is for display math.
    tex = TEX_TEMPLATE % ("\\[ %s \\]" % latex if not latex.strip().startswith("\\begin") else latex)
    with tempfile.TemporaryDirectory() as tmp:
        texfile = os.path.join(tmp, "eq.tex")
        with open(texfile, "w", encoding="utf-8") as f:
            f.write(tex)
        try:
            p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", texfile],
                               cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            return p.returncode == 0
        except Exception:
            return False
Use it during evaluation: for each beam, prefer one where latex_renders_ok() is True. If none render, choose beam with lowest edit distance.

(This is simple and effective. It requires pdflatex installed on the server.)

E. Inference endpoint (backend)
Add a FastAPI endpoint, e.g., in backend/main.py:

from fastapi import APIRouter, UploadFile, File
from recognition.predict import predict_image  # write wrapper

router = APIRouter()

@router.post("/recognize")
async def recognize(file: UploadFile = File(...), beam: int = 5):
    data = await file.read()
    # save to tmp
    tmp = "/tmp/rec.png"
    with open(tmp, "wb") as f:
        f.write(data)
    latex, beams = predict_image(tmp, num_beams=beam)   # predict_image returns best + list
    # try render and return render path
    ok = latex_renders_ok(latex)
    render_path = render_latex_to_png(latex)  # implement helper using pdflatex + convert
    return {"latex": latex, "render": render_path, "ok": ok, "candidates": beams}
predict_image should load the trained model and image processor and run generation with num_beams.

F. GUI integration (frontend)
Add “Recognize” button in side panel or equation modal: when clicked it sends the crop (or page with bbox) to /recognize.

When response returns:

Display latex in the LaTeX edit box,

Display rendered image in the render pane,

If ok==True show green check,

Allow user to accept & save or edit.

Also add a “Re-run recognition” button in the UI for a selected box.

G. RAG / persistence changes
When user clicks “Approve & Save”, save a record with:

{
  "paper_id":"xxx",
  "image_id":...,
  "bbox":[x0,y0,w,h],
  "latex":"...",
  "render": "path/to/png",
  "symbols": {...}  // optional: parsed glossary
}
Keep as jsonl per paper profile. Later create an embedding index (FAISS) over latex or combined text+latex for RAG.

Tests & sanity checks (how to verify)
Tokenizer sanity: run quick script to encode/decode a handful of LaTeX lines — ensure no junk tokens:

from transformers import PreTrainedTokenizerFast
tk = PreTrainedTokenizerFast(tokenizer_file='detector/tokenizer/latex_tokenizer.json', pad_token='[PAD]', bos_token='<s>', eos_token='</s>')
s = r"\frac{d}{dx}\sin x = \cos x"
print(tk.encode(s))
print(tk.decode(tk.encode(s)))
Small training run on 1k varied pairs with --freeze-encoder-epochs 2, --lr 1e-5, --batch_size 8, --num-workers 2. Check norm_edit reduces and examples become better.

Inference check:

POST a crop PNG to /recognize and check returned latex and rendered PNG.

Test with a few examples where ground truth exists.

GUI end-to-end check: load paper, select a detected box, click “Recognize”, verify LaTeX appears and render matches the image.

Short-term pragmatic roadmap (two-week sprint)
Day 1–2: Implement tokenizer training and integrate PreTrainedTokenizerFast into training script. Add skip_special_tokens=True decode & beam search.

Day 3–4: Expand synthetic generator to produce thousands of varied formulas. Create all_latex.txt for tokenizer training.

Day 5–6: Re-train model with new tokenizer & data (freeze 2 epochs, LR 1e-5, beams at eval). Log n-best outputs and test pdflatex validation on beams.

Day 7–8: Add recognize endpoint + render helper + GUI “Recognize” button.

Day 9–14: Iterate augmentations and tokenizer vocab size; add postprocessing and candidate beam validation via pdflatex or simple parser; add persistence for RAG.

Tools & libraries you’ll need
tokenizers (HuggingFace) and transformers (already used)

sentencepiece or tokenizers BPE trainer (I recommended tokenizers BPE)

pdflatex (MiKTeX / TeX Live) for rendering checks (or you can use matplotlib for simple inline math only)

ImageMagick or pdftoppm if you want PDF→PNG conversion

sympy (optional) for math parsing/validation (can be brittle; pdflatex is a robust validator)

If you want, I can:
Provide a drop-in train_latex_tokenizer.py patch (done above) and a wrapper to load it in training script.

Provide a small predict_image.py module and a FastAPI endpoint that your frontend can call.

Produce a render_latex_to_png helper and a beam-candidate selector that validates with pdflatex.

Tell me which piece you want me to produce first (tokenizer training code, synthetic-coco expansion, server inference endpoint, or GUI change), and I’ll give exact code you can paste into your repo.



