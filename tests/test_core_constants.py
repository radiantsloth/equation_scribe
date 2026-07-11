import sys
from pathlib import Path


CORE_SRC = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from equation_scribe import config as legacy_config
from equation_scribe_core.config import (
    DEFAULT_DPI,
    DEFAULT_PAGE_H_PX,
    DEFAULT_PAGE_W_PX,
    DESKEW_THRESHOLD_DEG,
    MAX_EQ_HEIGHT_FRAC,
    MAX_EQ_WIDTH_FRAC,
    MAX_PLACEMENT_ATTEMPTS,
    NON_OVERLAP_IOU,
    PAGE_HEIGHT_IN,
    PAGE_WIDTH_IN,
    ROTATION_AUG_MAX_ANGLE,
)


def test_core_constants_export_expected_shared_values():
    assert DEFAULT_DPI == 150
    assert PAGE_WIDTH_IN == 8.5
    assert PAGE_HEIGHT_IN == 11.0
    assert DEFAULT_PAGE_W_PX == int(PAGE_WIDTH_IN * DEFAULT_DPI)
    assert DEFAULT_PAGE_H_PX == int(PAGE_HEIGHT_IN * DEFAULT_DPI)
    assert ROTATION_AUG_MAX_ANGLE == 15
    assert DESKEW_THRESHOLD_DEG == 0.75
    assert MAX_EQ_WIDTH_FRAC == 0.6
    assert MAX_EQ_HEIGHT_FRAC == 0.25
    assert NON_OVERLAP_IOU == 0.0
    assert MAX_PLACEMENT_ATTEMPTS == 1000


def test_legacy_config_wrapper_reexports_core_constants():
    assert legacy_config.DEFAULT_DPI == DEFAULT_DPI
    assert legacy_config.PAGE_WIDTH_IN == PAGE_WIDTH_IN
    assert legacy_config.PAGE_HEIGHT_IN == PAGE_HEIGHT_IN
    assert legacy_config.DEFAULT_PAGE_W_PX == DEFAULT_PAGE_W_PX
    assert legacy_config.DEFAULT_PAGE_H_PX == DEFAULT_PAGE_H_PX
    assert legacy_config.ROTATION_AUG_MAX_ANGLE == ROTATION_AUG_MAX_ANGLE
    assert legacy_config.DESKEW_THRESHOLD_DEG == DESKEW_THRESHOLD_DEG
    assert legacy_config.MAX_EQ_WIDTH_FRAC == MAX_EQ_WIDTH_FRAC
    assert legacy_config.MAX_EQ_HEIGHT_FRAC == MAX_EQ_HEIGHT_FRAC
    assert legacy_config.NON_OVERLAP_IOU == NON_OVERLAP_IOU
    assert legacy_config.MAX_PLACEMENT_ATTEMPTS == MAX_PLACEMENT_ATTEMPTS
