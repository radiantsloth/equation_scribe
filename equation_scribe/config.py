"""Compatibility wrapper for shared stable constants.

The shared constants now live in ``equation_scribe_core.config.constants``.
This module re-exports them so existing imports keep working during the
migration.
"""

from pathlib import Path
import sys

try:
    from equation_scribe_core.config.constants import (
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
except ModuleNotFoundError:
    core_src = Path(__file__).resolve().parents[1] / "packages" / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    from equation_scribe_core.config.constants import (
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

__all__ = [
    "DEFAULT_DPI",
    "PAGE_WIDTH_IN",
    "PAGE_HEIGHT_IN",
    "DEFAULT_PAGE_W_PX",
    "DEFAULT_PAGE_H_PX",
    "ROTATION_AUG_MAX_ANGLE",
    "DESKEW_THRESHOLD_DEG",
    "MAX_EQ_WIDTH_FRAC",
    "MAX_EQ_HEIGHT_FRAC",
    "NON_OVERLAP_IOU",
    "MAX_PLACEMENT_ATTEMPTS",
]
