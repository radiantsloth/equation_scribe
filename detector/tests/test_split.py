import json
import tempfile
from pathlib import Path
from detector.split_coco_by_paper import split_coco_by_paper

def make_sample_coco(tmpdir):
    tmpdir = Path(tmpdir)
    images = [
        {"id": 1, "file_name": "paperA_page_0001.png", "width": 1000, "height": 1000},
        {"id": 2, "file_name": "paperA_page_0002.png", "width": 1000, "height": 1000},
        {"id": 3, "file_name": "paperB_page_0001.png", "width": 1000, "height": 1000},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id":1, "bbox":[10,10,100,50], "area":5000, "iscrowd":0},
        {"id": 2, "image_id": 3, "category_id":1, "bbox":[50,60,80,40], "area":3200, "iscrowd":0}
    ]
    coco = {"images": images, "annotations": annotations, "categories":[{"id":1,"name":"display"},{"id":2,"name":"inline"}]}
    p = tmpdir / "instances_all.json"
    json.dump(coco, open(p, "w"))
    return p

def test_split(tmp_path):
    coco = make_sample_coco(tmp_path)
    outdir = tmp_path / "out"
    t, v = split_coco_by_paper(coco, outdir, val_frac=0.5, seed=1)
    assert t.exists()
    assert v.exists()
    train = json.load(open(t))
    val = json.load(open(v))
    # ensure papers are grouped (no image appears in both)
    train_ids = set(img['id'] for img in train['images'])
    val_ids = set(img['id'] for img in val['images'])
    assert train_ids.intersection(val_ids) == set()
