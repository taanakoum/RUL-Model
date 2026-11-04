"""Image preparation from an :class:`ImageSpec` (PROJECT_PLAN.md §5.7).

Vendored from OmniFuser (``omnifuser/data.py``, commit e3b9985): ``IMAGE_MEAN``,
``IMAGE_STD``, ``load_photo`` (here ``load_image``: crop as top, left, bottom, right
fractions, then a bilinear resize) and the idea of ``ImageBank``. The crop, size and
standardization flag of each image type come from the adapter; this module only applies
them. Arrays are float32 (3, H, W), normalized with the ImageNet statistics that the
pretrained encoders expect.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from rulfuse.core.schema import ImageSpec

IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
STANDARD_LUMINANCE_MEAN = 0.5
STANDARD_LUMINANCE_SD = 0.2


def load_image(
    path: str | Path,
    height: int,
    width: int,
    crop: tuple[float, float, float, float] | None = None,
) -> Image.Image:
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
    """Rescale an RGB image in [0, 1] so its luminance has the given mean and sd (one
    affine map for all channels, then clipped). A flat image is set to the target mean."""
    rgb = np.asarray(rgb, dtype=np.float64)
    luminance = rgb @ np.array([0.299, 0.587, 0.114])
    spread = float(luminance.std())
    if spread < 1e-6:
        return np.full_like(rgb, mean, dtype=np.float32)
    out = (rgb - float(luminance.mean())) * (sd / spread) + mean
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepared_pixels(path: str | Path, spec: ImageSpec) -> np.ndarray:
    """(H, W, 3) float32 in [0, 1]: cropped, resized and (if the spec says so) standardized."""
    image = load_image(path, spec.height, spec.width, spec.crop)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return standardize_luminance(pixels) if spec.standardize else pixels


def normalize(pixels: np.ndarray) -> np.ndarray:
    """(H, W, 3) in [0, 1] -> (3, H, W) float32 ImageNet-normalized."""
    return ((pixels - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1).astype(np.float32, copy=False)


class ImageBank:
    """Decode each image once and keep its prepared pixels as uint8."""

    def __init__(self, paths: Sequence[str | Path], spec: ImageSpec) -> None:
        self.spec = spec
        self.paths = [Path(p) for p in dict.fromkeys(str(p) for p in paths)]
        self.index = {str(path): position for position, path in enumerate(self.paths)}
        self.pixels = np.empty((len(self.paths), spec.height, spec.width, 3), dtype=np.uint8)
        for position, path in enumerate(self.paths):
            prepared = prepared_pixels(path, spec)
            self.pixels[position] = np.round(prepared * 255.0).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.paths)

    def array(self, path: str | Path) -> np.ndarray:
        """(3, H, W) float32, ImageNet-normalized."""
        return normalize(self.pixels[self.index[str(Path(path))]].astype(np.float32) / 255.0)
