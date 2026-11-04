"""Command line: ``rulfuse <command>`` (PROJECT_PLAN.md §14.3). Stages 0-1 commands."""

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


def cmd_resources(_: argparse.Namespace) -> int:
    from rulfuse.resources import check_resources

    print(check_resources().describe())
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    from rulfuse.verify import verify_data

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
    from rulfuse.data.force_store import ForceStore, build_force_store
    from rulfuse.data.nonastreda import load_labels
    from rulfuse.resources import ResourcesBusy, check_resources

    try:
        status = check_resources(min_memory_mb=args.min_memory_mb)
    except ResourcesBusy as error:
        print(f"not started: {error}", file=sys.stderr)
        return 2
    print(f"resources: {status.describe()}")
    out = build_force_store(args.raw, args.out, load_labels(args.raw))
    store = ForceStore.open(out)
    print(json.dumps(store.meta, indent=1))
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
    from rulfuse.data.dataset import DatasetOptions, load_dataset
    from rulfuse.data.segment_store import SegmentStore, build_segment_store

    options = DatasetOptions(
        trim_rule=args.trim_rule,
        axis=args.axis,
        attach=args.attach,
        count_leading=args.count_leading,
        ttf_rule=args.ttf_rule,
    )
    ds = load_dataset(options, raw=args.raw, store_dir=args.force_store)
    out = build_segment_store(ds, args.out, force=args.rebuild)
    store = SegmentStore.open(out)
    print(f"segment store {store.signature} at {out}")
    print(json.dumps(store.meta["counts"], indent=1))
    return 0


def cmd_store_info(args: argparse.Namespace) -> int:
    from rulfuse.data.segment_store import SegmentStore

    roots = [args.path] if args.path else sorted((paths.interim_dir() / "segments").glob("*/"))
    for root in roots:
        store = SegmentStore.open(root)
        print(f"{store.signature}  {json.dumps(store.meta['payload']['options'])}")
        print(f"  {json.dumps(store.meta['counts'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rulfuse")
    sub = parser.add_subparsers(dest="command", required=True)

    segments = sub.add_parser("build-store", help="segment store for one set of data options")
    segments.add_argument("--raw", type=Path, default=paths.raw_dir())
    segments.add_argument("--force-store", type=Path, default=paths.force_store_dir())
    segments.add_argument("--out", type=Path, default=None, help="default: interim/segments/<sig>")
    segments.add_argument("--trim-rule", default="causal", choices=["causal", "omnifuser"])
    segments.add_argument("--axis", default="gap", choices=["gap", "recorded"])
    segments.add_argument("--attach", default="latest", choices=["latest", "every", "dedup"])
    segments.add_argument("--count-leading", action="store_true")
    segments.add_argument(
        "--ttf-rule",
        default="confirmed",
        choices=["confirmed", "literal", "wear110", "tool_label3", "dulled_onwards"],
    )
    segments.add_argument("--rebuild", action="store_true", help="rebuild even if it exists")
    segments.set_defaults(func=cmd_build_store)

    info = sub.add_parser("store-info", help="signature, options and counts of segment stores")
    info.add_argument("path", nargs="?", type=Path, default=None)
    info.set_defaults(func=cmd_store_info)

    sub.add_parser("resources", help="idle CPUs and free memory").set_defaults(func=cmd_resources)

    verify = sub.add_parser("verify-data", help="hashes, counts and gaps of the raw copy")
    verify.add_argument("--raw", type=Path, default=paths.raw_dir())
    verify.add_argument("--repo-data", type=Path, default=paths.REPO_ROOT / "Data")
    verify.add_argument(
        "--out", type=Path, default=paths.runs_dir() / "verify" / "verify-data.json"
    )
    verify.set_defaults(func=cmd_verify_data)

    store = sub.add_parser("build-force-store", help="read the MAT file once into a memmap store")
    store.add_argument("--raw", type=Path, default=paths.raw_dir())
    store.add_argument("--out", type=Path, default=paths.force_store_dir())
    store.add_argument("--min-memory-mb", type=float, default=1500.0)
    store.set_defaults(func=cmd_build_force_store)

    hub = sub.add_parser("hub-check", help="planned checkpoints exist / gated (metadata only)")
    hub.add_argument("--out", type=Path, default=paths.runs_dir() / "verify" / "hub-check.json")
    hub.set_defaults(func=cmd_hub_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
