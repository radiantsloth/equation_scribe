# tools/swap_formula_image_cols.py
import csv
from pathlib import Path
import sys

def swap(input_csv, out_tsv, header=True):
    in_path = Path(input_csv)
    out_path = Path(out_tsv)
    with in_path.open("r", encoding="utf-8", errors="replace") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        # detect header (simple heuristic)
        first = next(reader)
        # If first row appears to be header names
        is_header = False
        lower = [c.lower() for c in first]
        if any("formula" in c for c in lower) and any("image" in c for c in lower):
            is_header = True
        if is_header:
            # skip header and continue
            pass
        else:
            # first row is actual data; process it
            # treat first element as formula, second as image
            if len(first) >= 2:
                formula = first[0].strip()
                img = first[1].strip()
                fout.write(f"{img}\t{formula}\n")

        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            formula = row[0].strip()
            img = row[1].strip()
            if formula == "" or img == "":
                continue
            fout.write(f"{img}\t{formula}\n")

    print(f"Wrote swapped TSV to {out_path} (from {in_path})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    args = p.parse_args()
    swap(args.inp, args.out)
