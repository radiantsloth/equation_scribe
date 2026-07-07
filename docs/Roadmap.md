# The Roadmap

## Spiral 1: The "IEEE" Detector (Fixing the Foundation)

* Goal: Replace the buggy "lines-by-y" logic with a column-aware clustering algorithm. IEEE papers are 2-column; your current code merges them.

* Test: A unit test that takes a 2-column layout and asserts that equations on the left don't merge with text on the right.

## Spiral 2: The "Brain" (Model Integration)

* Goal: Swap your training script for a pre-trained SOTA model (LaTeX-OCR) to generate valid LaTeX immediately without training.

* Test: A script that takes a crop of an equation and prints the LaTeX string.

## Spiral 3: The "Loop" (GUI Integration)

* Goal: Wire the backend to auto-run Spiral 1 & 2 when a PDF is uploaded, populating the GUI with "Draft" boxes.

* Test: Upload a PDF in the UI and see bounding boxes appear automatically.

-----------------------
## Spiral 1 — Human-in-the-loop PDF Equation Scribe (SPIRAL1_ROADMAP)
## Summary (goal)

Build an interactive GUI & lightweight backend that lets humans verify and correct equations auto-detected in PDFs. Deliver a repeatable pipeline from PDF → heuristic detector → human in the loop editing → validated equations saved as JSONL profiles.

## What we implemented (completed)

* PDF rendering and metadata helpers (page images, pdf_to_px/px_to_pdf).

* Heuristic detector for equation candidates (tuned thresholds).

* Profile indexing (PROFILES_ROOT/index.json) and CLI registration (register_paper).

* Backend endpoints for:

- profile index, find-by-basename, list/save/update/delete equations,

- page image/meta and LaTeX validation.

Frontend (React + Konva) that supports:

drawing/moving/resizing boxes (corner + edge handles),

selecting “current boxes” and saved boxes,

LaTeX edit box with KaTeX preview,

Save / Update / Delete flows wired to backend,

Auto-load profile by PDF basename.

Storage safety: portalocker-backed index read/writes and backup history for profile writes.

Unit tests for PDF ingest & detection fixed and passing.

Remaining / polish items (recommended)

Delete confirmation + undo for deletions.

Checksum-based profile lookup (fallback to basename): compute & store PDF checksum to avoid missing profile when filename changed.

Auto-pan/zoom to saved box when selecting a saved box off-screen.

Larger improvements in autodetector (ML-based / multi-line grouping) to reduce human work in Spiral 2.

CI: pytest in GitHub Actions, and linting for frontend.

E2E tests for critical flows (open PDF → edit box → validate → save → verify).

How to proceed (next steps)

Prioritize: (1) checksum lookup + profile registration, (2) autodetector improvement, (3) sympy integration and symbol glossary.

For each item, create a small branch, add unit tests for the core logic, and produce a PR that CI runs.

File locations

Backend: equation_scribe_web/backend/

Frontend: equation_scribe_web/frontend/

Index & profiles: $PROFILES_ROOT/index.json and $PROFILES_ROOT/<paper_id>/equations.jsonl

Roadmap docs: equation_scribe_web/docs/

# Spiral 2

Spiral 2 Roadmap — Robust Automated Equation Detection & Recognition

Owner: Team Spangler
Repo: equation_scribe + equation_scribe_web
Doc: docs/SPIRAL2_ROADMAP.md
Primary goal: Produce a robust automated first pass extractor that detects equation locations in PDFs and converts equation images into high-quality LaTeX so the human-in-the-loop only needs to perform minor edits. Target: ~75% first-pass correctness (semantic/LaTeX-level).

High-level Objectives

Robust equation detection: Build a detector that reliably finds both display-style and inline equations in born-digital and scanned PDFs across a variety of journals and layouts.

Accurate equation recognition: Convert cropped equation images into correct LaTeX strings, with confidence scores and alternatives.

End-to-end pipeline: Combine detection+recognition into an inference pipeline (server endpoint) that writes high-quality candidates into PROFILES_ROOT/<paper>/equations.jsonl.

Human-in-the-loop UX: Frontend UI controls to re-detect / re-run recognition on any box, select alternatives, accept corrections, and mark examples for training.

Active learning: Capture corrected examples and bootstrap retraining to improve models continuously.

Evaluation & metrics: Define objective metrics (detection mAP, recognition sequence accuracy, semantic equivalence) and acceptance criteria.

Acceptance Criteria (what success looks like)

Detection:

mAP@0.5 ≥ 0.75 (display equations)

Inline equation recall ≥ 0.60 at IoU≥0.5

Recognition (end-to-end):

First-pass recognition (predicted LaTeX) ≥ 75% of cases judged semantically equivalent or exact-match (on a held-out validation set)

Human effort:

Average manual edit time per paper reduced by ≥ 50% vs current heuristic baseline

Robustness:

Works on both born-digital PDFs and moderate-quality scans (with OCR fallback)

UX:

Frontend supports redetect/rerecognize on a box and shows top-k hypotheses + confidence

Architecture Overview
PDF (pages)  -->  Detection (page image)  -->  Cropped boxes  \
                                                              --> Recognition (Im2LaTeX/Pix2Struct + LLM/SymPy postprocess)
PDF text-layer --> Text heuristics (fallback or ensemble)  --> Candidate boxes  /


Key components

Detector: object detection model (YOLOv8 / Detectron2 / Faster R-CNN) that outputs boxes with confidence and class (display, inline).

Recognizer: image-to-LaTeX model (Im2LaTeX-style encoder–decoder or Pix2Struct/Donut) + optional OCR-assisted hybrid and LLM-based normalization/validation.

Validator: SymPy/ANTLR to parse/validate LaTeX and optionally check semantic equivalence.

Ensembler: combine text-based heuristics with ML detector (union / weighted NMS).

Backend endpoints: new and updated endpoints to support full-paper autodetect and per-box re-detection.

Frontend UI: redetect button, alternatives list, confidence display, training toggle.

Data: Sources, Format, and Labeling
Sources

Synthetic LaTeX renders: render LaTeX snippets at multiple fonts, DPI, and noise levels (fast way to scale recognition data).

ArXiv LaTeX + rendered PDFs: parse/compile arXiv papers to extract real-world LaTeX and use mapping between source and PDF for ground truth (where available).

Im2LaTeX & public datasets: use Im2LaTeX-100k and other public datasets.

Manual labels: use the GUI to collect ground truth: (image crop, page_index, bbox_pdf, LaTeX). Target an initial labeled set: 3k–5k boxes for detection, 10k–30k labeled crops for recognition (initial).

MathPix (optional): use as a high-quality weak labeler if budget permits.

Storage format

Keep using PROFILES_ROOT/<paper_id>/equations.jsonl but extend each record with:

{
  "eq_uid": "<uid>",
  "paper_id": "MyPaper",
  "latex": "<predicted latex>",
  "confidence": 0.92,
  "alternatives": [{"latex":"...", "confidence":0.70}, ...],
  "validation": {
    "sympy_ok": true,
    "parse_error": null,
    "semantic_equivalence": true
  },
  "boxes": [{ "page": 3, "bbox_pdf": [x0,y0,x1,y1], "detector_score": 0.87 }],
  "source": "autodetect"  // or "manual"
}

Models & Tools (recommended)
Detection

Primary: YOLOv8 (Ultralytics) or Detectron2 (Faster R-CNN / Mask R-CNN)

Pros: fast training, good small-object handling (YOLOv8), strong tooling for augmentation

Train to detect display_equation, inline_equation

Optional segmentation: mask-based methods to get tight contours for cropping

Recognition

Primary open-source: Im2LaTeX (ResNet + Transformer decoder) or Pix2Struct/Donut-style models

Hybrid: Tesseract/PaddleOCR for plain text tokens + transformer that consumes image + OCR tokens

LLM postprocess: small LLM (local or API) to canonicalize LaTeX and fix syntax, aided by SymPy parse checks

Commercial fallback: MathPix API for bootstrapping or production-critical recognition

Validation

SymPy + antlr4-python3-runtime==4.11 for LaTeX parsing and semantic checks

Numerical test: compare symbolic expressions by substituting random numeric values for symbols

Backend API Design (new/updated endpoints)

Add or update these endpoints in backend/main.py:

Full-paper autodetect
POST /papers/{paper_id}/autodetect
Request: { "pdf_path": "<path or url>", "mode": "fast|robust", "min_confidence": 0.5 }
Response: { "task_id": "<id>", "status": "started" }  // optionally asynchronous


Runs detection over all pages, runs recognition per box, writes updated equations.jsonl with confidence & alternatives.

Per-box re-recognition (redetect)
POST /papers/{paper_id}/equations/{eq_uid}/redetect
Request: { "box": { "page": 1, "bbox_pdf": [x0,y0,x1,y1] }, "top_k": 3 }
Response: { "alternatives": [ { "latex": "...", "confidence": 0.95 }, ... ], "chosen": "<top>" }


Used by frontend “Redetect” button.

Find-by-checksum (optional)
GET /papers/find_by_checksum?checksum=<sha256>


Prefer checksum lookup for robust matching (add checksum when registering).

Frontend UX Requirements
Per-box UI

Redetect button: calls per-box re-recognition endpoint.

Alternatives dropdown: displays top-k hypotheses for quick selection.

Confidence bar: display detector + recognizer confidence; color-coded (green/yellow/red).

Accept / Reject: accept sets the latex, reject can mark as needs_review.

Add to training: checkbox — user corrections appended to PROFILES_ROOT/<paper>/human_corrections/ for retraining.

Batch Tools

Auto-accept high-confidence: Accept all boxes with combined confidence > threshold.

Review low-confidence: Show a filtered view for boxes needing human attention.

Active Learning & Model Retraining

Capture: Save each human correction as {image, bbox_pdf, page_index, gold_latex, metadata} in PROFILES_ROOT/<paper>/human_corrections/.

Pool: Collect corrections across papers. Periodically (e.g., weekly) prepare a training dataset.

Retrain: Fine-tune detector/recognizer using new corrections + synthetic data.

Deploy: Evaluate on holdout set; if metrics improve, publish new model to the inference endpoint.

Logging: Track stats of corrections per paper, false positives, common error classes.

Evaluation Plan and Tests
Detection tests

Measure mAP@0.5 and recall separately for display vs inline classes.

Unit tests for the postprocessing logic (NMS, ensemble rules, page-edge cases).

Recognition tests

Exact-match accuracy on synthetic+real test set.

SymPy semantic equivalence test:

parse ground-truth and predicted LaTeX

check sympy.simplify(pred - gold) == 0 or numeric evaluation on random inputs

Track top-k accuracy for beam search.

End-to-end tests

Run autodetect on a bank of 50 validation pages; measure percent of boxes with valid recognition (≥ 75% target).

Maintain an automated benchmark pipeline and report at PR time.

Data & Labeling Plan

Small initial labeled set — manually label 3k–5k boxes (mix of display/inline).

Use frontend GUI; add "label mode" that exports training records.

Synthetic augmentation — render LaTeX snippets with fonts, DPI, jitter, and noise.

Harvest pseudo-labels — use MathPix or existing models to generate weak labels, then human-verify key samples.

Annotation schema:

paper_id, page_index, bbox_pdf, latex_gold, source, notes

Store as newline JSON in PROFILES_ROOT/<paper>/human_corrections.jsonl

Milestones & Timeline (estimate)

All durations are approximate and assume 1–2 engineers + access to GPU.

Week 0 (Setup): Data pipeline, labelling tools, synthetic renderer, storage layout.

Weeks 1–3 (Detector baseline): label 3k boxes, train YOLOv8, integrate backend detector endpoint. → milestone 1

Weeks 3–6 (Recognizer baseline): train Im2LaTeX on synthetic + labeled crops; implement recognition service and postprocessing. → milestone 2

Weeks 6–8 (Integration & UI): full pipeline + frontend redetect, alternatives UI, store corrections. → milestone 3

Weeks 9–12 (Refinement & active learning): retrain, improve metrics, tackle edge cases. → milestone 4

Weeks 12+ (Production hardening): CI, monitoring, fallback integration (MathPix), large-scale data augmentation.

Total: ~2–3 months to a working 75% first-pass system (with iterative improvements thereafter).

Compute & infrastructure

Development: single GPU (RTX 2080/3080/4090) sufficient for experiments.

Training: multi-GPU (A100/V100) recommended to accelerate training on larger datasets.

Serving: inference GPUs for recognition; CPU-based detector for small-scale; containerize models (Docker + TorchServe or FastAPI).

Storage: object store for labeled dataset and models (S3 / on-prem NFS), PROFILES_ROOT for profiles and corrections.

Risks & Mitigations

Scans & noisy docs: OCR fallback (Tesseract/PaddleOCR) + augmentation in training reduces failure rate.

Ambiguity in LaTeX: LLM normalization + SymPy validation catches a lot; keep top-k alternatives and allow human selection.

Data labeling scale: Use synthetic pretraining + limited human labeling; consider paid labeling (Mathpix or annotation services) for bootstrapping.

Cost (commercial APIs): Keep MathPix optional; design system so it can be toggled as fallback.

CI & Monitoring

CI: add a GitHub Action that runs pytest, linting, and small smoke tests for backend endpoints (/papers/find_by_pdf, /papers/{paper_id}/page/0/meta) on PRs.

Model monitoring: Log per-paper detection/recognition confidence and human correction rates to monitor drift. Track metrics after each retrain.

Minimal APIs & Data Schemas (quick reference)

Equation record (jsonl):

{
  "eq_uid": "uuid",
  "paper_id": "paper_id",
  "latex": "\\frac{1}{2}mv^2",
  "confidence": 0.92,
  "alternatives": [{"latex": "...", "confidence": 0.4}],
  "validation": {"sympy_ok":true, "semantic_equivalence":true, "parse_error":null},
  "boxes": [{"page":3, "bbox_pdf":[x0,y0,x1,y1], "detector_score":0.88}],
  "source":"autodetect"
}


Endpoints:

POST /papers/{paper_id}/autodetect

POST /papers/{paper_id}/equations/{eq_uid}/redetect

GET /models/detector/status (optional)

GET /models/recognizer/status (optional)

Next concrete tasks (pick one to start)

Implement per-box redetect UI & endpoint (quick, valuable).

Labeling pipeline + 3k box labeling sprint (critical data).

YOLOv8 detector baseline training script & inference endpoint.

Recognition microservice: Im2LaTeX skeleton + LLM postprocess notebook.

If you want fast impact and to collect training data quickly, start with (1) Redetect button — it both improves UX and produces labeled pairs (model input + corrected LaTeX) that power retraining.

Appendix: Suggested repo file & folder additions
docs/SPIRAL2_ROADMAP.md       # this roadmap
detector/                     # detector training and inference scripts
recognizer/                   # recognition training/inference
models/                       # stored model manifests
scripts/render_latex.py       # create synthetic images
data/train/                   # labeled training examples
PROFILES_ROOT/<paper>/human_corrections.jsonl
backend/main.py               # new endpoints (autodetect, redetect)
frontend/src/api/client.ts    # new client helpers (redetect/findProfileByPdf)
frontend/src/App.tsx          # call redetect + handle alternatives


# Detector / Latex encoder roadmap 
1) If your detector is already reliably finding equations across the kinds of pages you care about, prioritize building a robust image→LaTeX recognizer next (because recognition quality depends heavily on good crops, and you already have a detector that mostly works).
2) If the detector is not yet robust to the variety of real papers you expect to process, invest in scaling up the detection dataset (more real scanned pages, better tiling/augmentation) in parallel with a small recognition baseline. Ultimately you’ll need both at production quality, but the immediate next step depends on the current detector reliability on real papers.

Below I give a concrete decision rubric and a detailed, practical plan you can act on (data, models, metrics, tests, integration, and a suggested incremental roadmap).

How to decide now (quick rubric)

Detector works well on real pages (visual check + high val mAP on real data)? → go build/improve the image→LaTeX model next.

Detector works only on synthetic / tiles or misses many real-paper equations? → prioritize more/better detection data and tiling logic first (so recognizer gets good crops), but start a small recognizer baseline in parallel.

Why: recognition needs clean, whole-equation crops. If the detector fails, recognizer training and evaluation will be noisy and slow.

A practical end-to-end plan (phased, prioritized)
Phase A — Minimal recognition baseline (short, get a proof-of-concept)

Goal: convert a correctly cropped equation image → LaTeX with reasonable accuracy.

Steps

Synthetic training data (fast): render lots of LaTeX formulas into images with the same DPI/scale as your detector crops (you already have render_latex.py and synthetic pipeline). Vary font, size, rotation, blur, JPEG compression and background/noise to match scanned pages.

Model: start with a proven image-to-sequence architecture:

CNN encoder (ResNet or EfficientNet) → Transformer decoder (standard img2latex / encoder-decoder architecture). The original im2latex or the more modern Transformer-based variants are good starting points.

Alternatives: “pix2struct” style or Vision-Transformer encoders + autoregressive decoder.

Loss & decoding: cross-entropy with teacher forcing; decode with beam search. Use byte-pair or tokenized LaTeX vocabulary (don’t treat entire LaTeX as single tokens).

Metrics: token-level error (BLEU / normalized Levenshtein), exact-match rate, and a structural metric (compare SymPy-parsed expression trees when possible). Keep a validation set of held-out synthetic formulas.

Quick eval: ensure the model can recover typical formulas (fractions, superscripts, subscripts, simple matrices). If it looks promising, proceed to Phase B.

Why this first: synthetic training is quick and gives you a working recognizer you can iterate on.

Phase B — Add real data and fine-tune (medium effort)

Goal: transfer from synthetic to real scanned crops.

Data

Collect a set of real equation crops from papers (scan or use PDFs). Label them with ground-truth LaTeX (human labeling or extract from sources where available). A few thousand pairs is an excellent start. Use your GUI for human-in-the-loop labeling.

Useful external data sources:

Im2LaTeX-100k (LaTeX source → PNG) — good for synthetic/real-looking.

CROHME (handwritten math) if you care about handwriting.

Kaggle math OCR, and existing equation datasets (search im2latex resources).

Augment real crops by noise, slight rotations, contrast.

Training

Fine-tune the synthetic-trained model on this real crop dataset.

Use data augmentation and mix synthetic/real batches to avoid catastrophic forgetting.

Metrics

Token error rate, string exact match %, and structural correctness:

Parse predicted LaTeX to a symbolic tree (SymPy or other LaTeX→AST) and compute tree-match or normalized edit distance.

Also measure Symbol Error Rate (SER) (like CER but for math symbols) and Expression Exact Match.

Why: recognizer trained on real crops will generalize to scanned papers.

Phase C — End-to-end integration & human-in-loop

Goal: detection -> recognition -> validation with GUI and human repairs.

Integration

Detection → Crop Normalization: after detection, crop with a consistent margin, correct skew/rotation, and normalize image size to recognizer input (keep aspect ratio). If rotated equations occur, deskew before recognition.

Recognizer scoring: return candidate LaTeX strings with confidence/beam hypotheses.

Validator LLM / symbolic check:

Try parsing the predicted LaTeX into SymPy. If SymPy parses, run small symbolic tests (simplify, differentiate) to ensure syntactic correctness.

If SymPy fails but confidence is high, present to human with rendered LaTeX (and the image).

Human-in-loop UI: show original crop, predicted LaTeX, rendered LaTeX, a “validate” button, and “edit” field. The user edits, LLM re-renders and re-checks parsing; final accepted LaTeX saved into paper profile.

Glossary / symbol definitions: keep a per-paper symbol table (user can define that E is electric field vs energy) and use it to help a symbolic engine or downstream SymPy operations.

Evaluation

End-to-end exact-match rate (crop → final accepted LaTeX) with human corrections counted separately.

Human time per page saved (goal: small corrections vs rewriting).

Model choices & libraries / concrete suggestions

Image→LaTeX model options

Im2LaTeX-style (best starting point): CNN encoder + Transformer decoder. Many open-source implementations exist. Train from scratch or fine-tune.

TrOCR / Donut / Pix2Struct: general OCR/image-to-text architectures. They are powerful, but may need heavy fine-tuning to handle mathematical structure. Donut (Doc VQA family) can be adapted to math, but Tokenization and math syntax require care.

Sequence-to-tree: advanced — you can use a decoder that emits LaTeX tokens but also produce an AST; harder but can improve structural correctness.

Decoding improvements

Beam search + length normalization.

Constrained decoding for LaTeX grammar (you can enforce bracket/command constraints).

Use token-level language model (LM) to re-rank beams for syntactic correctness.

Post-processing / error correction

Use an LLM or heuristic grammar-corrector to fix common LaTeX mistakes (unbalanced braces, misuse of \\ etc). But don’t fully rely on LLMs for syntactic-critical fixes — prefer deterministic syntax validators and parsers (SymPy, antlr).

Symbol parsing

Use sympy.parsing.latex.parse_latex (with ANTLR runtime set up) to check whether LaTeX can be parsed into math objects when possible. Also consider writing a custom lightweight LaTeX expression parser for common constructs to increase robustness.

Data & augmentation specifics

Synthetic rendering

Vary fonts (Computer Modern, Times, ArialMath), sizes, DPI (100–300), skew ±15°, blur, JPEG compression, background texture (paper grain), ink bleed, and occlusions.

Add line-height and nearby text context so recognizer can learn about inline vs display.

Real-data curation

Annotate diverse publishers (IEEE, Springer, arXiv/TeX source) and scans (different scanners/phone photos).

Label edge cases: multi-line equations, matrices, integrals, piecewise functions.

Evaluation metrics (be explicit)

Token-level metrics: BLEU or normalized Levenshtein between predicted and reference LaTeX.

Exact-match: percent of predictions that exactly match the ground truth LaTeX (after canonicalization/whitespace normalization).

Symbol Error Rate (SER): per-symbol substitution/insert/delete rate.

Structural correctness: parse both strings to expression trees and measure tree-edit distance or whether they are algebraically equivalent (using SymPy for algebraic expressions).

Human effort: average edits per equation in the UI — ultimate metric for a human-in-loop system.

How to organize work (parallel tracks)

If you have one or two engineers, a good split:

Week 1–2 (both tracks)

Detector: collect & label 200–500 real pages (or focus on hard negatives), run tile-inference and check errors.

Recognizer: create synthetic dataset matched to detector crops (50k–200k pairs) and train a baseline im2latex model for a few epochs.

Week 3–4

Detector: retrain with more real data, improve augmentation, tune tiling and inference.

Recognizer: fine-tune on small real labeled set (1k–3k), test on held-out.

Week 5–6

Integration: end-to-end tests on a few real papers, UI checks, measure human correction time.

Iterate on tricky cases (matrices, multi-line equations, inline vs display).

Practical tips & gotchas

Start small and measure — don’t try to perfect recognition for every LaTeX package at once. Start with a core subset (fractions, superscripts, subscripts, roots, operators, matrices) and expand.

Keep canonicalization rules — many LaTeX forms are equivalent; canonicalize before comparing (normalize differences like \frac{1}{2} vs inline 1/2 formatting).

Use beam search for the recognizer and present top-k to human annotator to reduce editing.

Error logging: record confusion cases and retrain targeted augmentations or collect more labeled examples for them.

Final recommendation for you right now

You said training looks good — that’s a great sign. Start the recognition work next, but keep a short parallel effort to expand detector coverage for “hard” real-paper cases.

Concretely:

Create a recognizer baseline trained on your synthetic crops (use your render pipeline to make a training set matching detector crop size and DPI).

Fine-tune on a small real set (500–2,000 labeled equation crops) drawn from papers you care about.

Integrate into your UI so the recognizer outputs a LaTeX candidate and rendered preview, and the human can approve or edit.

Iterate based on human edits: add corrected examples to your fine-tuning set.

If you want, I’ll:

give a ready-to-run im2latex training recipe (data format, model architecture, tokenization), or

produce a short script to tile pages, run inference with your trained detector, and stitch predicted boxes back to page coordinates for end-to-end tests, or

sketch the recognizer architecture and provide hyperparameters / training schedule.

Which do you want me to prepare first — (A) an im2latex starter recipe and code, or (B) a tile-inference + stitch script to verify page-level detection?