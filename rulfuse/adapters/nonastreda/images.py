"""Nonastreda image types and their preparation (PROJECT_PLAN.md §5.7).

* **tool:** rows 0.30-0.85 at full width (the OmniFuser photo probe's band containing the
  cutting edge in every photo), resized to 224 × 672;
* **chip:** resized to 448 × 448;
* **work:** resized to 224 × 224, then luminance-standardized per image: Stage 1 found that
  the workpiece photos' brightness and contrast step exactly at the ``tool_label`` stage
  changes (``docs/EDA_RUL.md`` §6), so global exposure must not reach the encoder.

The core applies these specs (:mod:`rulfuse.core.images`); only the values live here.
"""

from __future__ import annotations

from rulfuse.adapters.nonastreda.load import PHOTO_KINDS, photo_path
from rulfuse.core.schema import ImageSpec

TOOL_EDGE_BAND = (0.30, 0.0, 0.85, 1.0)

PHOTO_SPECS: dict[str, ImageSpec] = {
    "tool": ImageSpec(224, 672, TOOL_EDGE_BAND),
    "chip": ImageSpec(448, 448),
    "work": ImageSpec(224, 224, standardize=True),
}

__all__ = ["PHOTO_KINDS", "PHOTO_SPECS", "TOOL_EDGE_BAND", "photo_path"]
