"""Command line: ``rulfuse <command>`` (PROJECT_PLAN.md §14.3).

Dataset-specific commands (``verify-data``, ``build-force-store``) belong to the Nonastreda
adapter; everything else takes ``--dataset`` and works on the generic store.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rulfuse import paths


def _write_json(obj: object, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}")


def _nonastreda_options(args: argparse.Namespace) -> dict:
    return {
        "trim_rule": args.trim_rule,
        "axis": args.axis,
        "attach": args.attach,
        "count_leading": args.count_leading,
        "ttf_rule": args.ttf_rule,
        "raw": args.raw,
        "store_dir": args.force_store,
    }


def _dataset(args: argparse.Namespace):
    from rulfuse.adapters import load_dataset

    return load_dataset(args.dataset, **_nonastreda_options(args))


def _dataset_and_store(args: argparse.Namespace):
    """The adapter's dataset and its (built or reused) segment store."""
    from rulfuse.core.store import SegmentStore, build_segment_store

    ds = _dataset(args)
    store = SegmentStore.open(build_segment_store(ds))
    return ds, store


def cmd_resources(_: argparse.Namespace) -> int:
    from rulfuse.resources import check_resources

    print(check_resources().describe())
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    from rulfuse.adapters.nonastreda.verify import verify_data

    report = verify_data(args.raw, args.repo_data)
    _write_json(report, args.out)
    print(
        f"files expected {report['files_expected']}, missing {len(report['files_missing'])}, "
        f"mismatched {len(report['files_mismatched'])}; passes {report['passes']}; "
        f"repo Data/ compared {report['repo_data'].get('files_compared', 0)} files, "
        f"differing {len(report['repo_data'].get('differing', []))}; ok={report['ok']}"
    )
    return 0 if report["ok"] else 1


def cmd_build_force_store(args: argparse.Namespace) -> int:
    from rulfuse.adapters.nonastreda.force_store import ForceStore, build_force_store
    from rulfuse.adapters.nonastreda.load import load_labels
    from rulfuse.resources import ResourcesBusy, check_resources

    try:
        status = check_resources(min_memory_mb=args.min_memory_mb)
    except ResourcesBusy as error:
        print(f"not started: {error}", file=sys.stderr)
        return 2
    print(f"resources: {status.describe()}")
    out = build_force_store(args.raw, args.out, load_labels(args.raw))
    print(json.dumps(ForceStore.open(out).meta, indent=1))
    return 0


def cmd_hub_check(args: argparse.Namespace) -> int:
    from rulfuse.hub import check_hub

    rows = check_hub()
    _write_json(rows, args.out)
    for row in rows:
        state = "missing" if not row.get("exists") else f"gated={row.get('gated')}"
        print(f"{row['id']:9s} {row['repo']:45s} {state}")
    return 0 if all(row.get("exists") for row in rows) else 1


def cmd_build_store(args: argparse.Namespace) -> int:
    from rulfuse.core.store import SegmentStore, build_segment_store

    ds = _dataset(args)
    out = build_segment_store(ds, args.out, force=args.rebuild)
    store = SegmentStore.open(out)
    print(f"segment store {store.signature} at {out}")
    print(json.dumps(store.meta["counts"], indent=1))
    return 0


def cmd_store_info(args: argparse.Namespace) -> int:
    from rulfuse.core.store import SegmentStore

    if args.path:
        roots = [args.path]
    else:
        roots = sorted(p.parent for p in (paths.interim_dir() / "segments").glob("*/*/meta.json"))
    for root in roots:
        store = SegmentStore.open(root)
        options = store.meta["payload"]["provenance"].get("options", {})
        print(f"{store.dataset}/{store.signature}  {json.dumps(options)}")
        print(f"  {json.dumps(store.meta['counts'])}")
    return 0


def cmd_extract_features(args: argparse.Namespace) -> int:
    from rulfuse.features import cache, check, encoders

    ds, store = _dataset_and_store(args)
    print(f"store {store.dataset}/{store.signature}")
    if args.kind == "signal":
        encoder = encoders.series_encoder(args.encoder or "mantis8m", args.device)
        features = cache.extract_signal_features(
            store, ds.signals, encoder, args.features_root, chunk=args.chunk, limit=args.limit
        )
        print(f"cache {features.folder}: {int(features.done.sum())}/{len(features.done)} done")
        if features.complete and not args.skip_checks:
            checks = {
                "scale_check": check.scale_check(store, ds.signals, encoder),
                "determinism": check.determinism_check(store, ds.signals, encoder, features),
            }
            _write_json(checks, features.folder / "checks.json")
            print(json.dumps(checks, indent=1))
    else:
        encoder = encoders.image_encoder(args.encoder or "dinov2b", args.device)
        features = cache.extract_image_features(
            store, encoder, args.features_root, kinds=args.kinds, batch=args.batch, limit=args.limit
        )
        print(f"cache {features.folder}: {int(features.done.sum())}/{len(features.done)} done")
    return 0


def cmd_feature_check(args: argparse.Namespace) -> int:
    from rulfuse.features import cache
    from rulfuse.features.check import feature_check

    _, store = _dataset_and_store(args)
    signal = cache.SignalFeatures.open(
        cache.find_features(store, "signal", args.signal_encoder, args.features_root)
    )
    image = None
    if args.image_encoder != "none":
        image = cache.ImageFeatures.open(
            cache.find_features(store, "image", args.image_encoder, args.features_root)
        )
    checks_path = signal.folder / "checks.json"
    extra = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.is_file() else {}
    report = feature_check(store, signal, image, extra=extra)
    out = args.out or paths.runs_dir() / "feature-check" / f"{store.dataset}-{store.signature}.json"
    _write_json(report, out)
    print(json.dumps(report["decisions"], indent=1))
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from rulfuse.baselines import score_usage_baselines
    from rulfuse.core.folds import leave_one_unit_out

    _, store = _dataset_and_store(args)
    result = score_usage_baselines(store, leave_one_unit_out(store.unit_names))
    out = (
        args.out or paths.runs_dir() / "baselines" / f"usage-{store.dataset}-{store.signature}.json"
    )
    _write_json(result, out)
    for name, row in result["summary"].items():
        print(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return 0


def _add_dataset_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="nonastreda", choices=["nonastreda"])
    p.add_argument(
        "--raw", type=Path, default=None, help="default: $RULFUSE_DATA_ROOT/raw/<dataset>"
    )
    p.add_argument("--force-store", type=Path, default=None, help="default: interim/forces")
    p.add_argument("--trim-rule", default="causal", choices=["causal", "omnifuser"])
    p.add_argument("--axis", default="gap", choices=["gap", "recorded"])
    p.add_argument("--attach", default="latest", choices=["latest", "every", "dedup"])
    p.add_argument("--count-leading", action="store_true")
    p.add_argument(
        "--ttf-rule",
        default="confirmed",
        choices=["confirmed", "literal", "wear110", "tool_label3", "dulled_onwards"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rulfuse")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-store", help="segment store for one set of dataset options")
    _add_dataset_options(p)
    p.add_argument(
        "--out", type=Path, default=None, help="default: interim/segments/<dataset>/<sig>"
    )
    p.add_argument("--rebuild", action="store_true", help="rebuild even if it exists")
    p.set_defaults(func=cmd_build_store)

    p = sub.add_parser("store-info", help="signature, options and counts of segment stores")
    p.add_argument("path", nargs="?", type=Path, default=None)
    p.set_defaults(func=cmd_store_info)

    p = sub.add_parser("extract-features", help="encode segments or images into a cache")
    _add_dataset_options(p)
    p.add_argument("--kind", required=True, choices=["signal", "image"])
    p.add_argument(
        "--encoder",
        default=None,
        help="signal: mantis8m (default) | projection; image: dinov2b (default) | pixelstats",
    )
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--kinds", nargs="*", default=None, help="image types (default: all)")
    p.add_argument("--chunk", type=int, default=64, help="segments per encoder call")
    p.add_argument("--batch", type=int, default=16, help="images per encoder call")
    p.add_argument(
        "--limit", type=int, default=None, help="encode at most this many now (smoke tests)"
    )
    p.add_argument("--skip-checks", action="store_true", help="no scale or determinism check")
    p.add_argument("--features-root", type=Path, default=None, help="default: interim/features")
    p.set_defaults(func=cmd_extract_features)

    p = sub.add_parser("feature-check", help="Stage-3 go/no-go, readouts and leakage probe")
    _add_dataset_options(p)
    p.add_argument("--signal-encoder", default="mantis8m")
    p.add_argument("--image-encoder", default="dinov2b", help="or 'none'")
    p.add_argument("--features-root", type=Path, default=None, help="default: interim/features")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_feature_check)

    p = sub.add_parser("baselines", help="usage-only baselines B0-B2 (seconds)")
    _add_dataset_options(p)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_baselines)

    sub.add_parser("resources", help="idle CPUs and free memory").set_defaults(func=cmd_resources)

    p = sub.add_parser("verify-data", help="Nonastreda: hashes, counts and gaps of the raw copy")
    p.add_argument("--raw", type=Path, default=paths.raw_dir())
    p.add_argument("--repo-data", type=Path, default=paths.REPO_ROOT / "Data")
    p.add_argument("--out", type=Path, default=paths.runs_dir() / "verify" / "verify-data.json")
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser(
        "build-force-store", help="Nonastreda: read the MAT file once into a memmap store"
    )
    p.add_argument("--raw", type=Path, default=paths.raw_dir())
    p.add_argument("--out", type=Path, default=paths.force_store_dir())
    p.add_argument("--min-memory-mb", type=float, default=1500.0)
    p.set_defaults(func=cmd_build_force_store)

    p = sub.add_parser("hub-check", help="planned checkpoints exist / gated (metadata only)")
    p.add_argument("--out", type=Path, default=paths.runs_dir() / "verify" / "hub-check.json")
    p.set_defaults(func=cmd_hub_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
