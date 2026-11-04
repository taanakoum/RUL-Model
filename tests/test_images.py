"""Image preparation: generic functions in the core, Nonastreda specs in the adapter (§5.7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rulfuse import paths
from rulfuse.adapters.nonastreda.images import PHOTO_SPECS
from rulfuse.core.images import (
    IMAGE_MEAN,
    IMAGE_STD,
    ImageBank,
    load_image,
    normalize,
    prepared_pixels,
    standardize_luminance,
)


def _banded_photo(path: Path, height: int = 100, width: int = 300) -> None:
    """Black above row 30, white rows 30-84, grey below: the tool crop keeps only white."""
    array = np.zeros((height, width, 3), np.uint8)
    array[30:85] = 255
    array[85:] = 128
    Image.fromarray(array).save(path)


def test_tool_crop_keeps_the_edge_band(tmp_path: Path):
    _banded_photo(tmp_path / "t.png")
    image = load_image(tmp_path / "t.png", 20, 60, PHOTO_SPECS["tool"].crop)
    pixels = np.asarray(image)
    assert pixels.shape == (20, 60, 3)
    assert pixels.min() >= 250  # only the white band (rows 30-85) survives the crop
    with pytest.raises(ValueError):
        load_image(tmp_path / "t.png", 10, 10, (0.5, 0.0, 0.4, 1.0))


def test_prepared_sizes_per_kind(tmp_path: Path):
    _banded_photo(tmp_path / "p.png")
    for spec in PHOTO_SPECS.values():
        assert prepared_pixels(tmp_path / "p.png", spec).shape == (spec.height, spec.width, 3)


def test_luminance_standardization_removes_exposure():
    rng = np.random.default_rng(0)
    texture = rng.uniform(0.3, 0.7, size=(64, 64, 1)).repeat(3, axis=2)
    dark, bright = 0.4 * texture, np.clip(texture + 0.25, 0, 1)
    for image in (texture, dark, bright):
        out = standardize_luminance(image)
        luminance = out @ np.array([0.299, 0.587, 0.114])
        assert luminance.mean() == pytest.approx(0.5, abs=0.01)
        assert luminance.std() == pytest.approx(0.2, abs=0.01)
    # The same texture under two exposures becomes (nearly) the same image.
    np.testing.assert_allclose(
        standardize_luminance(dark), standardize_luminance(texture), atol=0.02
    )
    assert np.all(standardize_luminance(np.full((4, 4, 3), 0.9)) == 0.5)


def test_normalize_and_bank(tmp_path: Path):
    raw = tmp_path
    (raw / "work").mkdir()
    _banded_photo(raw / "work" / "T1R1B1.png")
    path = raw / "work" / "T1R1B1.png"
    bank = ImageBank([path, path], PHOTO_SPECS["work"])
    assert len(bank) == 1
    array = bank.array(path)
    assert array.shape == (3, 224, 224) and array.dtype == np.float32
    expected = normalize(prepared_pixels(path, PHOTO_SPECS["work"]))
    np.testing.assert_allclose(array, expected, atol=1 / 255 / IMAGE_STD.min() + 1e-6)
    np.testing.assert_allclose(normalize(np.tile(IMAGE_MEAN, (2, 2, 1))), 0, atol=1e-6)


@pytest.mark.data
def test_real_workpiece_photos_are_standardized(metas):
    for meta in metas[::40]:
        pixels = prepared_pixels(
            paths.raw_dir() / "work" / f"{meta.record_id}.png", PHOTO_SPECS["work"]
        )
        luminance = pixels @ np.array([0.299, 0.587, 0.114])
        assert luminance.mean() == pytest.approx(0.5, abs=0.03)
        assert luminance.std() == pytest.approx(0.2, abs=0.03)
    tool = prepared_pixels(
        paths.raw_dir() / "tool" / f"{metas[0].record_id}.jpg", PHOTO_SPECS["tool"]
    )
    assert tool.shape == (224, 672, 3)
