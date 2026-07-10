"""Shared models for Equation Scribe core."""

from .index import PaperIndex, PaperIndexEntry
from .paper_profiles import BBox, Box, EquationRecord

__all__ = [
    "BBox",
    "Box",
    "EquationRecord",
    "PaperIndex",
    "PaperIndexEntry",
]
