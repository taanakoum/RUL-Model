"""Photo preparation (PROJECT_PLAN.md §5.7).

Vendored from OmniFuser (``omnifuser/data.py``, commit e3b9985): ``IMAGE_MEAN``,
``IMAGE_STD``, ``load_photo`` (crop given as top, left, bottom, right fractions, then a
bilinear resize) and the idea of ``ImageBank`` (decode each photo once). OmniFuser's bank
returns torch tensors; this one returns numpy arrays, because torch arrives only in Stage 3.

Per photo type:

* **tool:** rows 0.30-0.85 at full width (the OmniFuser photo probe's band containing the
  cutting edge in every photo), resized to 224 × 672;
* **chip:** resized to 448 × 448 (224 × 224 is the Gate-1 alternative);
* **work:** resized to 224 × 224, then *luminance-standardized per image* to a fixed mean
  and spread. Stage 1 found that the workpiece photos' brightness and contrast step
  exactly at the ``tool_label`` stage changes (``docs/EDA_RUL.md`` §6), so global
  exposure must not reach the encoder.

Arrays are float32 (3, H, W), normalized with the ImageNet statistics that the pretrained
encoders expect.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rulfuse.data.nonastreda import photo_path

IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
Crop = tuple[float, float, float, float]  # top, left, bottom, right, as fractions
TOOL_EDGE_BAND: Crop = (0.30, 0.0, 0.85, 1.0)
STANDARD_LUMINANCE_MEAN = 0.5
STANDARD_LUMINANCE_SD = 0.2


@dataclass(frozen=True)
class PhotoSpec:
    height: int
    width: int
    crop: Crop | None = None
    standardize: bool = False


PHOTO_SPECS: dict[str, PhotoSpec] = {
    "tool": PhotoSpec(224, 672, TOOL_EDGE_BAND),
    "chip": PhotoSpec(448, 448),
    "work": PhotoSpec(224, 224, standardize=True),
}


def load_photo(path: str | Path, height: int, width: int, crop: Crop | None = None) -> Image.Image:
    """The RGB photo, cropped to ``crop`` if given, resized to ``width`` x ``height``
    (vendored from OmniFuser)."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    if crop is not None:
        top, left, bottom, right = crop
        if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
            raise ValueError(f"invalid crop {crop}")
        image = image.crop(
            (
                round(left * image.width),
                round(top * image.height),
                round(right * image.width),
                round(bottom * image.height),
            )
        )
    return image.resize((width, height), Image.Resampling.BILINEAR)


def standardize_luminance(
    rgb: np.ndarray,
    mean: float = STANDARD_LUMINANCE_MEAN,
    sd: float = STANDARD_LUMINANCE_SD,
) -> np.ndarray:
    """Rescale an RGB image in [0, 1] so that its luminance has the given mean and sd.

    One affine map is applied to all three channels (it keeps colour ratios), then the
    result is clipped to [0, 1]. A flat image is set to the target mean.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    luminance = rgb @ np.array([0.299, 0.587, 0.114])
    spread = float(luminance.std())
    if spread < 1e-6:
        return np.full_like(rgb, mean, dtype=np.float32)
    out = (rgb - float(luminance.mean())) * (sd / spread) + mean
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepared_pixels(path: str | Path, kind: str) -> np.ndarray:
    """(H, W, 3) float32 in [0, 1]: cropped, resized and (workpiece) standardized."""
    if kind not in PHOTO_SPECS:
        raise ValueError(f"photo kind must be one of {sorted(PHOTO_SPECS)}")
    spec = PHOTO_SPECS[kind]
    image = load_photo(path, spec.height, spec.width, spec.crop)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return standardize_luminance(pixels) if spec.standardize else pixels


def normalize(pixels: np.ndarray) -> np.ndarray:
    """(H, W, 3) in [0, 1] -> (3, H, W) float32 ImageNet-normalized."""
    return ((pixels - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1).astype(np.float32, copy=False)


class ImageBank:
    """Decode every photo once and keep its prepared pixels as uint8."""

    def __init__(self, raw: str | Path, record_ids: Iterable[str], kind: str) -> None:
        self.kind = kind
        self.spec = PHOTO_SPECS[kind]
        ids = list(dict.fromkeys(record_ids))
        self.index = {record_id: position for position, record_id in enumerate(ids)}
        self.pixels = np.empty((len(ids), self.spec.height, self.spec.width, 3), dtype=np.uint8)
        for position, record_id in enumerate(ids):
            prepared = prepared_pixels(photo_path(raw, record_id, kind), kind)
            self.pixels[position] = np.round(prepared * 255.0).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.index)

    def array(self, record_id: str) -> np.ndarray:
        """(3, H, W) float32, ImageNet-normalized."""
        return normalize(self.pixels[self.index[record_id]].astype(np.float32) / 255.0)
