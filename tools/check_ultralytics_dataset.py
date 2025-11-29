# save as tools/check_ultralytics_dataset.py and run: python tools/check_ultralytics_dataset.py
import json, traceback
from pathlib import Path

yaml_path = Path("equation_scribe/detector/detector.yaml").resolve()
print("Using YAML:", yaml_path)

try:
    # ultralytics may export check_det_dataset under different locations depending on version
    from ultralytics.data.utils import check_det_dataset
except Exception as e:
    print("Could not import ultralytics.check_det_dataset:", e)
    raise

try:
    info = check_det_dataset(str(yaml_path))
    print("check_det_dataset returned (summary):")
    # print keys & small summary
    for k, v in info.items():
        if isinstance(v, (list, dict)):
            print(f"  {k}: {type(v)} (len={len(v)})")
        else:
            print(f"  {k}: {v}")
    # If it contains 'train' or 'val' with resolved lists, print first 10 entries
    for split in ("train", "val"):
        print(f"\nResolved {split} images (first 10):")
        arr = info.get(split) or info.get(f"{split}_images") or []
        for x in arr[:10]:
            print("  ", x)
except Exception as e:
    print("check_det_dataset raised exception:")
    traceback.print_exc()
