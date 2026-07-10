"""Typed models for the paper profile index.

The index file keeps lightweight metadata about each paper profile directory and
provides a reverse lookup from PDF basename to ``paper_id``. These typed models
give the rest of the codebase a shared vocabulary before storage logic is
centralized in a later migration step.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PaperIndexEntry(BaseModel):
    """Metadata for one paper in ``index.json``."""

    paper_id: str = Field(..., description="Stable identifier for the paper profile.")
    pdf_basename: str = Field(
        ...,
        description="Lowercased PDF file basename used for reverse lookup.",
    )
    profiles_dir: str = Field(
        ...,
        description="Directory name under the profiles root that stores this paper.",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="UTC ISO timestamp for initial registration, when available.",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="UTC ISO timestamp for the most recent index update.",
    )
    num_equations: Optional[int] = Field(
        default=None,
        description="Optional cached count of equations stored for this paper.",
    )


class PaperIndex(BaseModel):
    """Top-level structure stored in ``index.json``."""

    version: int = Field(1, description="Index schema version.")
    papers: Dict[str, PaperIndexEntry] = Field(
        default_factory=dict,
        description="Map of paper_id to its stored metadata entry.",
    )
    by_pdf_basename: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of lowercased PDF basename to paper_id.",
    )
