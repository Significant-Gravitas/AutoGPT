"""Backfill an empty ``StoreListingVersion.subHeading`` with a generated CTA.

Marketplace listings now require a short call-to-action sub-heading, but rows
predating that rule can hold an empty string, which leaves preview cards to
fall back on the full description.

A dry run (the default) generates a one-line CTA per listing from its name and
description via ``gpt-4o-mini`` and writes ``id,name,description,
proposed_sub_heading`` to the CSV; it writes nothing to the database. Review
the CSV, editing or deleting rows as needed, then ``--apply`` writes that CSV's
``proposed_sub_heading`` values to the rows whose subHeading is still empty.
``--apply`` generates nothing.

Usage::

    poetry run python scripts/backfill_store_sub_headings.py            # dry run
    poetry run python scripts/backfill_store_sub_headings.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

DEFAULT_CSV_PATH = Path("store_sub_heading_backfill.csv")
# Spreadsheet apps evaluate cells opening with these (CWE-1236).
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
# Matches the API validator, which strips all whitespace; btrim() trims only spaces.
BLANK_SUB_HEADING = "\"subHeading\" !~ '\\S'"

MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 40
# One stuck call otherwise inherits the OpenAI client's 600s default (x2 retries).
CALL_TIMEOUT_SECONDS = 30.0
RATE_LIMIT_DELAY_SECONDS = 0.2

SYSTEM_PROMPT = (
    "You write one-line marketplace taglines for automation agents. "
    "Given an agent's name and description, reply with a single "
    "call-to-action line of at most {max_length} characters saying what the "
    "agent does for the user. Start with a verb, name the outcome rather "
    "than the mechanism, and use sentence case with no trailing period. "
    "Example: 'Find decision-makers at any company in seconds'. "
    "Output only the line."
)


async def main(
    limit: int | None,
    include_unapproved: bool,
    csv_path: Path,
    apply: bool,
) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    # Imported lazily so --help works without the backend env being loaded.
    from backend.api.features.store.model import SUB_HEADING_MAX_LENGTH
    from backend.data.db import connect, disconnect

    if apply:
        reviewed = read_reviewed_csv(csv_path, SUB_HEADING_MAX_LENGTH)
        await connect()
        try:
            written = await persist_sub_headings(reviewed)
        finally:
            await disconnect()
        print(
            f"Updated {written} of {len(reviewed)} listing version(s) from {csv_path}."
        )
        return 0

    await connect()
    try:
        rows = await find_listings_without_sub_heading(limit, include_unapproved)
        print(
            f"{len(rows)} listing version(s) with an empty subHeading"
            f"{'' if include_unapproved else ' (APPROVED only)'}."
        )
        if not rows:
            return 0

        proposals = await generate_sub_headings(rows, SUB_HEADING_MAX_LENGTH)
        write_csv(csv_path, proposals)
        print(f"Wrote {len(proposals)} proposal(s) to {csv_path}")

        for row_id, name, _description, proposed in proposals[:5]:
            print(f"  {row_id}  {name!r}\n    -> {proposed!r}")

        print(
            f"\nDry run: nothing written. Review {csv_path}, then re-run with --apply."
        )
        return 0
    finally:
        await disconnect()


async def find_listings_without_sub_heading(
    limit: int | None, include_unapproved: bool
) -> list[tuple[str, str, str]]:
    from prisma import get_client

    # Prisma has no "empty once trimmed" string filter.
    status_clause = (
        "" if include_unapproved else "AND \"submissionStatus\" = 'APPROVED'"
    )
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    rows = await get_client().query_raw(
        f"""
        SELECT id, name, description
        FROM platform."StoreListingVersion"
        WHERE {BLANK_SUB_HEADING} AND "isDeleted" = false
        {status_clause}
        ORDER BY "createdAt" DESC
        {limit_clause}
        """
    )
    return [(r["id"], r["name"], r["description"]) for r in rows]


async def generate_sub_headings(
    rows: list[tuple[str, str, str]], max_length: int
) -> list[tuple[str, str, str, str]]:
    from backend.util.clients import get_openai_client

    client = get_openai_client()
    if client is None:
        raise SystemExit("No OpenAI client configured (OPENAI_API_KEY unset)")

    system_prompt = SYSTEM_PROMPT.format(max_length=max_length)
    proposals: list[tuple[str, str, str, str]] = []
    for index, (row_id, name, description) in enumerate(rows, start=1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Name: {name}\nDescription: {description}",
                        },
                    ],
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            print(f"  [{index}/{len(rows)}] timed out on {row_id}", file=sys.stderr)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad row must not end the run
            print(f"  [{index}/{len(rows)}] failed on {row_id}: {exc}", file=sys.stderr)
            continue

        proposed = clean_sub_heading(
            response.choices[0].message.content or "", max_length
        )
        if proposed:
            proposals.append((row_id, name, description, proposed))
        else:
            print(
                f"  [{index}/{len(rows)}] empty generation for {row_id}",
                file=sys.stderr,
            )

        await asyncio.sleep(RATE_LIMIT_DELAY_SECONDS)

    return proposals


def clean_sub_heading(raw: str, max_length: int) -> str:
    """Strip the quoting and trailing period models add, then enforce the cap."""
    line = raw.strip().split("\n")[0].strip().strip('"').strip("'").rstrip(".").strip()
    if len(line) <= max_length:
        return line
    # Truncating mid-word would ship a broken line; drop the row for review instead.
    return ""


def write_csv(path: Path, proposals: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "name", "description", "proposed_sub_heading"])
        writer.writerows(
            (row_id, *(escape_formula(cell) for cell in cells))
            for row_id, *cells in proposals
        )


def read_reviewed_csv(path: Path, max_length: int) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(f"{path} not found: run without --apply first to create it")
    reviewed: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_id = row["id"].strip()
            proposed = unescape_formula(row["proposed_sub_heading"]).strip()
            if not proposed or len(proposed) > max_length:
                print(f"  skipping {row_id}: {proposed!r} is empty or too long")
                continue
            reviewed.append((row_id, proposed))
    return reviewed


def escape_formula(cell: str) -> str:
    return "'" + cell if cell.startswith(FORMULA_PREFIXES) else cell


def unescape_formula(cell: str) -> str:
    return (
        cell[1:] if cell[:1] == "'" and cell[1:].startswith(FORMULA_PREFIXES) else cell
    )


async def persist_sub_headings(reviewed: list[tuple[str, str]]) -> int:
    from prisma import get_client

    client = get_client()
    written = 0
    for row_id, proposed in reviewed:
        # The emptiness re-check keeps a concurrent edit from being overwritten.
        written += await client.execute_raw(
            f"""
            UPDATE platform."StoreListingVersion"
            SET "subHeading" = $2
            WHERE id = $1 AND {BLANK_SUB_HEADING}
            """,
            row_id,
            proposed,
        )
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Dry run only.")
    parser.add_argument(
        "--include-unapproved",
        action="store_true",
        help="Dry run only: also cover DRAFT/PENDING/REJECTED versions, which fail "
        "validation on edit.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Written by a dry run; read by --apply.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the reviewed CSV's proposed_sub_heading values. Generates nothing.",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(main(args.limit, args.include_unapproved, args.csv, args.apply))
    )
