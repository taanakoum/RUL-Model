"""``rulfuse hub-check``: do the planned pretrained checkpoints exist, and are they gated?

PROJECT_PLAN.md §6.2 and Stage 0. Uses only Hugging Face metadata (no weights are
downloaded). Needs the ``hub`` extra and network access.
"""

from __future__ import annotations

CHECKPOINTS: dict[str, str] = {
    "MA": "paris-noah/Mantis-8M",
    "MO-base": "AutonLab/MOMENT-1-base",
    "MO-large": "AutonLab/MOMENT-1-large",
    "CH": "amazon/chronos-t5-small",
    "AST": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "D2-S": "facebook/dinov2-small",
    "D2-B": "facebook/dinov2-base",
    "SG2": "google/siglip2-base-patch16-224",
    "D3-S": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "D3-B": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


def check_hub(checkpoints: dict[str, str] | None = None) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    for key, repo in (checkpoints or CHECKPOINTS).items():
        row: dict[str, object] = {"id": key, "repo": repo}
        try:
            info = api.model_info(repo)
            row.update(
                exists=True,
                gated=info.gated or False,
                private=bool(info.private),
                sha=info.sha,
                last_modified=str(info.last_modified) if info.last_modified else None,
                library=info.library_name,
                license=(info.card_data or {}).get("license") if info.card_data else None,
            )
        except Exception as error:  # noqa: BLE001 - report every failure kind as a row
            row.update(exists=False, error=f"{type(error).__name__}: {error}"[:300])
        rows.append(row)
    return rows
