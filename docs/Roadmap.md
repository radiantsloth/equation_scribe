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