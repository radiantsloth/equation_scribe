import json
from pathlib import Path

from equation_scribe.detector.data_prep import convert_from_profiles
from equation_scribe.detector.data_prep_coco import convert_profiles_to_coco, find_equations_jsonl_files


def _write_profile(path: Path, *, paper_id: str = "paper-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "eq_uid": "eq-1",
                "paper_id": paper_id,
                "latex": "x+y",
                "notes": "",
                "boxes": [{"page": 0, "bbox_pdf": [10.0, 20.0, 30.0, 40.0]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_convert_from_profiles_accepts_single_jsonl_file(tmp_path: Path):
    profile_path = tmp_path / "paper-1" / "equations.jsonl"
    _write_profile(profile_path)
    out_path = tmp_path / "instances.json"

    convert_from_profiles(profile_path, out_annotations_path=out_path)

    coco = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 20.0, 20.0]


def test_find_equations_jsonl_files_and_convert_profiles_root(tmp_path: Path):
    profile_path = tmp_path / "profiles" / "paper-1" / "equations.jsonl"
    _write_profile(profile_path)
    out_path = tmp_path / "instances_all.json"

    files = find_equations_jsonl_files(tmp_path / "profiles")
    assert files == [profile_path]

    convert_profiles_to_coco(tmp_path / "profiles", out_path)

    coco = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
