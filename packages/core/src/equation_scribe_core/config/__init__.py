"""Configuration primitives for Equation Scribe core."""

from .constants import (
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
from .settings import (
    DEFAULT_PAPERS_ROOT,
    DEFAULT_PROFILES_ROOT,
    PAPERS_ROOT_ENV,
    PROFILES_ROOT_ENV,
    RuntimeSettings,
    get_runtime_settings,
)

__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_PAGE_H_PX",
    "DEFAULT_PAGE_W_PX",
    "DEFAULT_PAPERS_ROOT",
    "DEFAULT_PROFILES_ROOT",
    "DESKEW_THRESHOLD_DEG",
    "MAX_EQ_HEIGHT_FRAC",
    "MAX_EQ_WIDTH_FRAC",
    "MAX_PLACEMENT_ATTEMPTS",
    "NON_OVERLAP_IOU",
    "PAPERS_ROOT_ENV",
    "PAGE_HEIGHT_IN",
    "PAGE_WIDTH_IN",
    "PROFILES_ROOT_ENV",
    "ROTATION_AUG_MAX_ANGLE",
    "RuntimeSettings",
    "get_runtime_settings",
]
