"""Controlled sequential preparation of persisted Kazakhstan screening cells.

This is deliberately a local operator tool, not a public API endpoint. Start
with --dry-run; a pilot requires an explicit --limit 3.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Sequence

from fastapi import HTTPException

# Allow `python scripts/prepare_national_grid.py` from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes_monitoring import _perform_analysis
from app.services import monitoring_service, national_grid_service
from app.services.kazakhstan_grid import DEFAULT_GRID_VERSION, configured_grid_cells, grid_plan_summary

RATE_LIMIT_SECONDS = 1.5
PROVENANCE_VERSION = "national-grid-preparation-v1"


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("must be true or false")
    return normalized == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare persisted Kazakhstan national screening cells sequentially.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--grid-version", default=DEFAULT_GRID_VERSION)
    parser.add_argument("--limit", type=int, default=None, help="Maximum selected cells; use 3 for the initial pilot.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without DB writes or external calls.")
    parser.add_argument("--resume", action="store_true", help="Skip cells that already have a ready observation.")
    parser.add_argument("--cell-id", action="append", default=[], help="Repeat to select individual cells.")
    parser.add_argument("--enrich", type=_bool, default=False, metavar="true|false")
    parser.add_argument("--force", action="store_true", help="Explicitly replace a ready observation; use only after review.")
    return parser


def planned_cells(grid_version: str, cell_ids: Sequence[str], limit: int | None) -> list[dict]:
    cells = configured_grid_cells(grid_version)
    if cell_ids:
        requested = set(cell_ids)
        cells = [cell for cell in cells if cell["cell_id"] in requested]
        missing = requested - {cell["cell_id"] for cell in cells}
        if missing:
            raise ValueError(f"Unknown or out-of-bound cell IDs: {', '.join(sorted(missing))}")
    return cells[:limit] if limit is not None else cells


def run(args: argparse.Namespace) -> int:
    if args.year < 2017:
        raise ValueError("year must be 2017 or later")
    cells = planned_cells(args.grid_version, args.cell_id, args.limit)
    summary = grid_plan_summary(args.grid_version)
    print(f"Grid {args.grid_version}: {summary['inside_kazakhstan_count']} cells inside Kazakhstan from {summary['candidate_count']} bounding-box candidates.")
    print(f"Planned for {args.year:04d}-{args.month:02d}: {len(cells)} cells; sequential rate limit {RATE_LIMIT_SECONDS:.1f}s between requests.")
    for cell in cells:
        print(f"  {cell['cell_id']}: {cell['latitude']:.3f}, {cell['longitude']:.3f}")
    if args.dry_run:
        print("DRY RUN: zero database writes, zero Earth Engine calls, and zero NASA POWER calls were made.")
        return 0
    if args.force:
        print("WARNING: --force may replace ready observations for the selected cells only.")

    persisted = {cell["cell_id"]: cell for cell in national_grid_service.ensure_grid_cells(args.grid_version)}
    created = skipped = missing = failed = 0
    for index, plan in enumerate(cells):
        cell = persisted[plan["cell_id"]]
        existing = national_grid_service.grid_results(args.year, args.month, args.grid_version, has_observation=True)
        existing_ids = {item["cell_id"] for item in existing if item["status"] == "ready"}
        if args.resume and cell["cell_id"] in existing_ids and not args.force:
            skipped += 1
            print(f"SKIP ready {cell['cell_id']}")
            continue
        try:
            location = monitoring_service.get_location(cell["location_id"])
            _perform_analysis(
                location, date(args.year, args.month, 1), args.enrich,
                {"source_type": "national_grid", "grid_version": args.grid_version, "config_version": PROVENANCE_VERSION},
                args.month, force=args.force,
            )
            national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, args.month, "ready", provenance={"config_version": PROVENANCE_VERSION})
            created += 1
            print(f"READY {cell['cell_id']}")
        except HTTPException as error:
            status = "missing" if error.status_code == 404 else "failed"
            national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, args.month, status, str(error.detail), {"config_version": PROVENANCE_VERSION})
            missing += status == "missing"
            failed += status == "failed"
            print(f"{status.upper()} {cell['cell_id']}: {error.detail}")
        except Exception as error:  # keep a single failed cell from aborting the batch
            national_grid_service.record_status(args.grid_version, cell["cell_id"], args.year, args.month, "failed", str(error), {"config_version": PROVENANCE_VERSION})
            failed += 1
            print(f"FAILED {cell['cell_id']}: {error}")
        if index < len(cells) - 1:
            time.sleep(RATE_LIMIT_SECONDS)
    print(f"Summary: ready={created}, skipped={skipped}, missing={missing}, failed={failed}.")
    return 0 if failed == 0 else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
