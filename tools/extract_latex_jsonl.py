#!/usr/bin/env python3
"""
Extract LaTeX strings (one per line) from a recognition_pairs jsonl file.

Usage:
    python tools/extract_latex_jsonl.py --jsonl detector/data/recognition_pairs/all.jsonl --out detector/data/recognition_pairs/all_latex.txt
"""
import argparse
import json
from pathlib import Path

def extract(jsonl_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # ignore invalid JSON lines
                continue

            # Accept numerous possible keys that may contain latex:
            # 'latex', 'text', 'tex', 'label', 'target'
            latex = None
            for key in ("latex", "text", "tex", "label", "target"):
                val = rec.get(key)
                if val:
                    latex = val
                    break

            # if found, normalize and write
            if latex:
                # replace newline chars, collapse multiple spaces, strip
                s = str(latex).replace("\r", " ").replace("\n", " ").strip()
                if s:
                    f_out.write(s + "\n")
                    count += 1

    print(f"Wrote {count} LaTeX lines to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to recognition_pairs/all.jsonl")
    parser.add_argument("--out", required=True, help="Path to output all_latex.txt")
    args = parser.parse_args()
    extract(Path(args.jsonl), Path(args.out))
