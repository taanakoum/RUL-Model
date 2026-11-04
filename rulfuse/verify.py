"""``rulfuse verify-data``: the raw copy is complete and unchanged (PROJECT_PLAN.md §15, P1).

Checks every file of the publisher manifest (``MANIFEST.json``, written by Project 14's
fetcher with the publisher's sha256 values) except the unused spectrogram folder, then the
labels, then that the repository's ``Data/`` folder holds the same files.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rulfuse.data.force_store import sha256_file
from rulfuse.data.nonastreda import PHOTO_KINDS, load_labels, metas_by_tool, photo_path
from rulfuse.data.timeline import run_gaps

SKIPPED_PREFIXES = ("spec/",)


TEXT_SUFFIXES = (".csv", ".json", ".txt")


def _same_text(a: Path, b: Path) -> bool:
    """True if two text files differ at most in their line endings."""
    if a.suffix.lower() not in TEXT_SUFFIXES:
        return False
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def _hash_all(paths: list[Path], threads: int = 4) -> dict[Path, str]:
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return dict(zip(paths, pool.map(sha256_file, paths), strict=True))


def verify_data(
    raw: str | Path,
    repo_data: str | Path | None = None,
    threads: int = 4,
    expected_passes: int = 512,
) -> dict:
    root = Path(raw)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        rel: info for rel, info in manifest["files"].items() if not rel.startswith(SKIPPED_PREFIXES)
    }
    present = [root / rel for rel in expected if (root / rel).is_file()]
    hashes = _hash_all(present, threads)
    missing, mismatched = [], []
    for rel, info in expected.items():
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif path.stat().st_size != info["size"] or hashes[path] != info["sha256"]:
            mismatched.append(rel)

    metas = load_labels(root)
    by_tool = metas_by_tool(metas)
    photos_missing = {
        kind: [m.record_id for m in metas if not photo_path(root, m.record_id, kind).is_file()]
        for kind in PHOTO_KINDS
    }
    gaps = {
        tool: {"leading": list(g.leading), "interior": list(g.interior)}
        for tool, tool_metas in by_tool.items()
        for g in [run_gaps(tool_metas)]
    }

    repo: dict[str, object] = {"checked": False}
    if repo_data is not None and Path(repo_data).is_dir():
        repo_root = Path(repo_data)
        candidates = [rel for rel in expected if (repo_root / rel).is_file()]
        repo_hashes = _hash_all([repo_root / rel for rel in candidates], threads)
        unequal = [
            rel for rel in candidates if repo_hashes[repo_root / rel] != expected[rel]["sha256"]
        ]
        # A text file whose only change is CRLF line endings holds the same data.
        line_endings_only = [rel for rel in unequal if _same_text(repo_root / rel, root / rel)]
        repo = {
            "checked": True,
            "path": str(repo_root),
            "files_compared": len(candidates),
            "differing": [rel for rel in unequal if rel not in line_endings_only],
            "line_endings_only": line_endings_only,
        }

    report = {
        "raw": str(root),
        "citation": manifest.get("citation"),
        "files_expected": len(expected),
        "files_missing": missing,
        "files_mismatched": mismatched,
        "passes": len(metas),
        "passes_per_tool": {tool: len(items) for tool, items in by_tool.items()},
        "photos_missing": photos_missing,
        "run_gaps": gaps,
        "repo_data": repo,
    }
    report["ok"] = bool(
        not missing
        and not mismatched
        and len(metas) == expected_passes
        and not any(photos_missing.values())
        and (not repo["checked"] or not repo["differing"])
    )
    return report
