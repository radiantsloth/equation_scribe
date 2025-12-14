# tools/convert_im2latex.py
import argparse
from pathlib import Path
import json
import shutil

def convert(mapping_file: Path, images_root: Path, out_jsonl: Path, img_rel_prefix="im2latex"):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with mapping_file.open("r", encoding="utf-8") as f, out_jsonl.open("w", encoding="utf-8") as out:
        for line in f:
            line=line.strip()
            if not line:
                continue
            # accept "imgpath TAB latex" or "imgpath SPACE latex" formats; adjust if needed
            parts=line.split("\t")
            if len(parts) == 1:
                parts=line.split(" ", 1)
            if len(parts) < 2:
                continue
            img_rel, latex = parts[0].strip(), parts[1].strip()
            img_path = images_root / img_rel
            if not img_path.exists():
                # try absolute path or skip
                # print("Missing:", img_path)
                continue
            rec = {
                "image": str(img_path),
                "text": latex,
                "paper_id": "im2latex",
                "meta": {}
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print("Wrote", n, "records to", out_jsonl)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", required=True, help="Tab-separated mapping file image[TAB]latex")
    p.add_argument("--images-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    convert(Path(args.mapping), Path(args.images_root), Path(args.out))
