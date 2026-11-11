"""Frozen encoders (PROJECT_PLAN.md D16, §6.2): Mantis-8M for every signal channel and
DINOv2-B/14 for every image type, plus two weight-free stand-ins for tests and the toy.

Vendored from Project 14 (``pdmfm/encoders.py``): ``hf_revision``, ``MantisEncoder`` and
``DinoV2Encoder`` (class token + patch grid average-pooled to a small grid). Changes: the
DINOv2 checkpoint is ``facebook/dinov2-base`` (768-d) instead of -small; one grid (2 × 4,
9 tokens) for every image type; pixels come prepared from :mod:`rulfuse.core.images`; and
networks can be injected, so the wrappers are tested with tiny random networks offline.

Every encoder has ``signature()``, part of each feature-cache key: features of different
checkpoints, revisions or settings are never mixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MANTIS_REPO = "paris-noah/Mantis-8M"
DINOV2_REPO = "facebook/dinov2-base"
WINDOW = 512  # Mantis's input length


WEIGHT_PATTERNS = ["*.json", "*.safetensors"]  # what slurm/prefetch_weights.sh downloads


def hf_revision(repo: str) -> str | None:
    """Commit hash of the cached snapshot of ``repo`` (None if no weights are cached).

    Only the files the loaders read are cached (``WEIGHT_PATTERNS``). huggingface_hub 1.x
    raises ``IncompleteSnapshotError`` for an unfiltered local lookup when other repo files
    (README, images) are absent, so the lookup uses the same patterns, and falls back to the
    cache's ``refs/main`` with at least one weight file in its snapshot."""
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo, local_files_only=True, allow_patterns=WEIGHT_PATTERNS))
        if any(path.glob("*.safetensors")):
            return path.name
    except Exception:  # noqa: BLE001 - not cached, offline, or another hub version
        pass
    try:
        from huggingface_hub import constants

        folder = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        revision = (folder / "refs" / "main").read_text(encoding="utf-8").strip()
        snapshot = folder / "snapshots" / revision
        return revision if any(snapshot.glob("*.safetensors")) else None
    except Exception:  # noqa: BLE001 - no cache, or no hub installed
        return None


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


# ----------------------------------------------------------------------------- series


class MantisEncoder:
    """(m, 512) windows -> (m, 256) frozen Mantis-8M embeddings."""

    name = "mantis8m"
    dim = 256
    length = WINDOW
    repo = MANTIS_REPO

    def __init__(self, device: str = "auto", batch_size: int = 512, network=None) -> None:
        from mantis.architecture import MantisV1  # pyright: ignore[reportMissingImports]
        from mantis.trainer import MantisTrainer  # pyright: ignore[reportMissingImports]

        self.device = resolve_device(device)
        self.pretrained = network is None
        if network is None:
            network = MantisV1(device=self.device).from_pretrained(self.repo)
        self.model = MantisTrainer(device=self.device, network=network)
        self.batch_size = batch_size

    def signature(self) -> dict:
        return {
            "encoder": self.name,
            "repo": self.repo if self.pretrained else "random-init",
            "revision": hf_revision(self.repo) if self.pretrained else None,
            "dim": self.dim,
            "length": self.length,
        }

    def encode(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            out = self.model.transform(x[:, None, :], batch_size=self.batch_size)
        return np.asarray(out, dtype=np.float32)


class ProjectionEncoder:
    """Weight-free stand-in: a fixed random projection of each z-normalized window, then
    tanh. Scale-invariant like a normalized foundation encoder; for tests and the toy."""

    name = "projection"
    length = WINDOW

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 1 / np.sqrt(self.length), (self.length, dim))

    def signature(self) -> dict:
        return {"encoder": self.name, "dim": self.dim, "seed": self.seed, "length": self.length}

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
        return np.tanh(z @ self.weights).astype(np.float32)


# ----------------------------------------------------------------------------- images

GRID = (2, 4)


class DinoV2Encoder:
    """(n, 3, H, W) ImageNet-normalized pixels -> (n, 9, 768): the class token and the patch
    grid average-pooled to 2 × 4."""

    name = "dinov2b"
    repo = DINOV2_REPO
    patch = 14

    def __init__(self, device: str = "auto", batch_size: int = 8, model=None) -> None:
        import torch

        self.device = resolve_device(device)
        self.pretrained = model is None
        if model is None:
            from transformers import AutoModel  # pyright: ignore[reportMissingImports]

            model = AutoModel.from_pretrained(self.repo)
        torch.nn.Module.to(model, self.device)
        self.model = model.eval()
        self.dim = int(model.config.hidden_size)
        self.batch_size = batch_size
        self.n_tokens = 1 + GRID[0] * GRID[1]

    def signature(self) -> dict:
        return {
            "encoder": self.name,
            "repo": self.repo if self.pretrained else "random-init",
            "revision": hf_revision(self.repo) if self.pretrained else None,
            "dim": self.dim,
            "grid": list(GRID),
        }

    def encode(self, pixels: np.ndarray) -> np.ndarray:
        import torch

        pixels = np.asarray(pixels, dtype=np.float32)
        height, width = pixels.shape[2:]
        grid_h, grid_w = height // self.patch, width // self.patch
        out = []
        with torch.no_grad():
            for start in range(0, len(pixels), self.batch_size):
                batch = torch.from_numpy(pixels[start : start + self.batch_size]).to(self.device)
                hidden = self.model(pixel_values=batch).last_hidden_state
                cls, patches = hidden[:, :1], hidden[:, 1 : 1 + grid_h * grid_w]
                patches = patches.reshape(len(batch), grid_h, grid_w, -1).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.adaptive_avg_pool2d(patches, GRID)
                pooled = pooled.flatten(2).transpose(1, 2)
                out.append(torch.cat([cls, pooled], dim=1).float().cpu().numpy())
        return np.concatenate(out)


class PixelStatsEncoder:
    """Weight-free stand-in: token 0 is the whole image's per-channel mean and sd, tokens
    1-8 the same over a 2 × 4 grid (6 values each)."""

    name = "pixelstats"
    dim = 6
    n_tokens = 1 + GRID[0] * GRID[1]

    def signature(self) -> dict:
        return {"encoder": self.name, "dim": self.dim, "grid": list(GRID)}

    def encode(self, pixels: np.ndarray) -> np.ndarray:
        x = np.asarray(pixels, dtype=np.float64)
        n, _, height, width = x.shape
        tokens = [np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1)]
        rows = np.array_split(np.arange(height), GRID[0])
        cols = np.array_split(np.arange(width), GRID[1])
        for r in rows:
            for c in cols:
                cell = x[:, :, r[0] : r[-1] + 1, c[0] : c[-1] + 1]
                tokens.append(np.concatenate([cell.mean(axis=(2, 3)), cell.std(axis=(2, 3))], 1))
        return np.stack(tokens, axis=1).astype(np.float32)


def series_encoder(name: str, device: str = "auto"):
    if name == "mantis8m":
        return MantisEncoder(device)
    if name == "projection":
        return ProjectionEncoder()
    raise ValueError(f"unknown series encoder {name!r}")


def image_encoder(name: str, device: str = "auto"):
    if name == "dinov2b":
        return DinoV2Encoder(device)
    if name == "pixelstats":
        return PixelStatsEncoder()
    raise ValueError(f"unknown image encoder {name!r}")
