# save as check_tiles_exist.py and run: python check_tiles_exist.py
import json, os, sys
from pathlib import Path

coco_path = Path("detector/data/annotations/instances_tiles_train.json")
if not coco_path.exists():
    print("COCO not found:", coco_path); sys.exit(1)

j = json.load(coco_path.open("r", encoding="utf-8"))
print("Images in JSON:", len(j.get("images", [])))
print("Annotations:", len(j.get("annotations", [])))

dataset_root = Path("detector/data").resolve()
missing = []
for im in j.get("images", []):
    fn = im["file_name"]
    img_path = dataset_root / fn  # the path Ultralytics will look for
    if not img_path.exists():
        missing.append((fn, str(img_path)))
        if len(missing) <= 20:
            print("MISSING:", fn, "->", img_path)
print("Total missing files:", len(missing))
print("Example existing test (first 5):")
for im in j.get("images", [])[:5]:
    print(im["file_name"], "exists?", (dataset_root / im["file_name"]).exists())
