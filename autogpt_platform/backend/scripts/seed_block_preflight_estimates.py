"""Standalone seeder for block_preflight_estimates.json.

Runs the same aggregation as ``GET /api/admin/blocks/cost-estimates`` against
``DATABASE_URL`` via Prisma, then writes the result to
``backend/data/block_preflight_estimates.json``. Use this to seed the file
before the admin endpoint is deployed (chicken-and-egg) or to refresh from a
machine with read-only prod credentials.

Usage::

    DATABASE_URL=postgresql://... \\
    poetry run python scripts/seed_block_preflight_estimates.py --days 7

If ``--dry-run`` is set, prints the JSON to stdout without touching the file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_BACKEND = Path(__file__).resolve().parent.parent
ESTIMATES_PATH = REPO_BACKEND / "backend" / "data" / "block_preflight_estimates.json"


async def main(days: int, min_samples: int, write: bool) -> None:
    if days <= 0:
        raise SystemExit("--days must be a positive integer")
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    # Imported lazily so --help works without the backend env being loaded.
    from backend.data.block_cost_analytics import compute_block_cost_estimates
    from backend.data.db import connect, disconnect

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    await connect()
    try:
        rows = await compute_block_cost_estimates(
            start=start, end=end, min_samples=min_samples
        )
    finally:
        await disconnect()

    estimates: dict[str, dict[str, object]] = {
        r.block_id: {
            "block_name": r.block_name,
            "cost_type": r.cost_type,
            "samples": r.samples,
            "mean_credits": r.mean_credits,
        }
        for r in rows
    }

    payload = {
        "version": 1,
        "generated_at": end.isoformat(),
        "source_window_days": days,
        "estimates": estimates,
    }
    serialised = json.dumps(payload, indent=2) + "\n"

    print(
        f"Aggregated {len(estimates)} dynamic-cost blocks over {days}d "
        f"with min_samples={min_samples}."
    )

    if write:
        ESTIMATES_PATH.write_text(serialised, encoding="utf-8")
        print(f"Wrote {ESTIMATES_PATH}")
    else:
        print(serialised)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=int(os.environ.get("BLOCK_PREFLIGHT_MIN_SAMPLES", "10")),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the JSON to stdout instead of writing the file.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.days, args.min_samples, write=not args.dry_run))
