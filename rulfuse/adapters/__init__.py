"""Dataset adapters (PROJECT_PLAN.md D18): each turns one dataset into the generic schema.

The core never imports this package; the command line and scripts do.
"""

from __future__ import annotations

from rulfuse.core.schema import Dataset

ADAPTERS = ("nonastreda",)


def load_dataset(name: str, **options: object) -> Dataset:
    if name == "nonastreda":
        from rulfuse.adapters.nonastreda import NonastredaOptions, load

        raw = options.pop("raw", None)
        store_dir = options.pop("store_dir", None)
        segment_s = float(options.pop("segment_s", None) or 10.0)  # type: ignore[arg-type]
        return load(
            NonastredaOptions(**options),  # type: ignore[arg-type]
            raw=raw,  # type: ignore[arg-type]
            store_dir=store_dir,  # type: ignore[arg-type]
            segment_s=segment_s,
        )
    raise ValueError(f"unknown dataset {name!r}; adapters: {ADAPTERS}")
