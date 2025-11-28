#!/usr/bin/env python3
"""
detector/synctex_extractor.py

Given a LaTeX source file (main.tex), this script:
- compiles it with pdflatex -synctex=1 (writes main.pdf and main.synctex.gz),
- finds display math regions (\begin{equation}, \[...\], $$...$$),
- for each region, calls synctex to find the PDF page and coordinates for the start line,
- returns a JSONL file with entries: {paper_id, page, bbox_pdf, latex}.

Usage:
  python detector/synctex_extractor.py --tex path/to/main.tex --pdf path/to/main.pdf --out detector/data/synctex_pairs.jsonl

Notes:
- Works best for papers with a TeX source available (arXiv).
- You may need to include the full paper preamble; the script compiles the provided .tex file as-is.
- The mapping uses the start line of the block as SyncTeX anchor.
"""
from pathlib import Path
import re
import subprocess
import argparse
import json
import tempfile
import shutil
import sys

DISPLAY_BEGIN_RE = re.compile(r'\\\\begin\\{(equation|align|align\\*|equation\\*|gather|multline)\\}')
DISPLAY_END_RE = re.compile(r'\\\\end\\{(equation|align|align\\*|equation\\*|gather|multline)\\}')
DOLLAR_DISPLAY_RE = re.compile(r'\\$\\$(.*?)\\$\\$', re.S)
BRACKET_RE = re.compile(r'\\\\\\[(.*?)\\\\\\]', re.S)

def find_display_regions(tex_path: Path):
    """
    Parse tex file line-by-line and extract display math regions.
    Return list of (start_line, end_line, latex_text).
    """
    regions = []
    with tex_path.open('r', encoding='utf-8', errors='ignore') as fh:
        lines = fh.readlines()

    # 1) find \begin{...} ... \end{...}
    i = 0
    while i < len(lines):
        m = DISPLAY_BEGIN_RE.search(lines[i])
        if m:
            start = i
            # find matching \end
            j = i
            depth = 0
            while j < len(lines):
                if DISPLAY_BEGIN_RE.search(lines[j]):
                    depth += 1
                if DISPLAY_END_RE.search(lines[j]):
                    depth -= 1
                    if depth <= 0:
                        end = j
                        break
                j += 1
            else:
                # not found; break
                i += 1
                continue
            latex = ''.join(lines[start:end+1])
            regions.append((start+1, end+1, latex))
            i = end + 1
            continue
        # check for \[ ... \]
        if '\\[' in lines[i]:
            start = i
            j = i
            found = False
            buf = []
            while j < len(lines):
                buf.append(lines[j])
                if '\\]' in lines[j]:
                    found = True
                    break
                j += 1
            if found:
                latex = ''.join(buf)
                regions.append((start+1, j+1, latex))
                i = j + 1
                continue
        # check for $$ ... $$
        if '$$' in lines[i]:
            start = i
            j = i
            buf = []
            cnt = lines[i].count('$$')
            # if odd count, we opened
            if cnt % 2 == 1:
                buf.append(lines[i])
                j = i + 1
                while j < len(lines):
                    buf.append(lines[j])
                    if lines[j].count('$$') % 2 == 1:
                        break
                    j += 1
                latex = ''.join(buf)
                regions.append((start+1, j+1, latex))
                i = j + 1
                continue
        i += 1
    return regions

def compile_with_synctex(tex_path: Path, workdir: Path):
    cmd = ['pdflatex', '-synctex=1', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(workdir), str(tex_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def synctex_view(pdf_path: Path, tex_path: Path, line: int):
    """
    Call synctex to map source line -> pdf coordinates.
    Returns a dict with page and boxes if found, or None.
    """
    synctex_cmd = shutil.which('synctex') or shutil.which('synctex.exe')
    if synctex_cmd is None:
        raise RuntimeError('synctex binary not found on PATH.')
    # synctex view -i <line>:<col>:<source.tex> -o <file.pdf>
    query = f"{line}:1:{str(tex_path)}"
    cmd = [synctex_cmd, 'view', '-i', query, '-o', str(pdf_path)]
    p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out = p.stdout + p.stderr
    # parse output: look for 'Output:' or 'Result' lines. Format can vary; we'll try to find 'Page:' and coords
    # Example synctex output lines: 'Output:Page: 5; x: 123.4; y: 456.7; h: ...'
    page = None
    x = y = None
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith('page:') or 'page:' in line.lower():
            # naive parse
            parts = line.replace(';', ' ').replace(':', ' : ').split()
            # find page number after 'page :'
            try:
                idx = parts.index('page') if 'page' in parts else (parts.index('Page') if 'Page' in parts else -1)
                if idx >= 0:
                    page = int(parts[idx+2])
            except Exception:
                pass
        # fallback: some implementations print 'Output: x y h v page n'
        m = re.search(r'Page\\s*[:=]?\\s*(\\d+)', line, flags=re.I)
        if m:
            page = int(m.group(1))
        # coords (x,y)
        m2 = re.search(r'x\\s*[:=]?\\s*([0-9.+-]+)', line, flags=re.I)
        m3 = re.search(r'y\\s*[:=]?\\s*([0-9.+-]+)', line, flags=re.I)
        if m2:
            try:
                x = float(m2.group(1))
            except: pass
        if m3:
            try:
                y = float(m3.group(1))
            except: pass
    if page is None:
        return None
    # synctex often returns an (x,y) in points; we return what we have
    return {"page": page-1 if page>0 else 0, "x": x, "y": y, "raw": out}

def extract_synctex_pairs(tex_path: Path, out_jsonl: Path, workdir: Path = None):
    tex_path = Path(tex_path)
    pdf_candidate = tex_path.with_suffix('.pdf')
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) if workdir is None else Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        # compile (copy tex into work dir to avoid overwriting project files)
        compile_with_synctex(tex_path, work)
        pdf_path = work / tex_path.name.replace('.tex', '.pdf')
        if not pdf_path.exists():
            # fallback: if user provided pdf, use it
            if pdf_candidate.exists():
                pdf_path = pdf_candidate
            else:
                raise RuntimeError('PDF not found after compilation.')

        regions = find_display_regions(tex_path)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, 'w', encoding='utf-8') as fh:
            for (start_line, end_line, latex) in regions:
                # use start_line as anchor
                mapping = synctex_view(pdf_path, tex_path, start_line)
                if mapping is None:
                    print(f'No synctex mapping for lines {start_line}-{end_line}; skipping')
                    continue
                rec = {
                    "latex": latex,
                    "page": mapping.get("page"),
                    "synctex_x": mapping.get("x"),
                    "synctex_y": mapping.get("y"),
                    "pdf_path": str(pdf_path),
                    "start_line": start_line,
                    "end_line": end_line,
                    "raw_synctex": mapping.get("raw")
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + '\\n')
        print(f'Wrote synctex pairs to {out_jsonl}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tex', required=True, help='Path to main .tex file')
    p.add_argument('--out', required=True, help='Output JSONL with extracted pairs')
    args = p.parse_args()
    extract_synctex_pairs(Path(args.tex), Path(args.out))

if __name__ == '__main__':
    main()
