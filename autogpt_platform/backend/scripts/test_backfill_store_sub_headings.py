"""Tests for the store sub-heading backfill script's CSV review round trip."""

import csv
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import scripts.backfill_store_sub_headings as backfill


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (
            '=HYPERLINK("https://evil.example","x")',
            '\'=HYPERLINK("https://evil.example","x")',
        ),
        ("+1+1", "'+1+1"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\t=1", "'\t=1"),
        ("Find leads in seconds", "Find leads in seconds"),
    ],
)
def test_write_csv_neutralises_formula_cells(tmp_path: Path, cell: str, expected: str):
    path = tmp_path / "out.csv"
    backfill.write_csv(path, [("id-1", cell, cell, cell)])

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[1] == ["id-1", expected, expected, expected]


def test_reviewed_csv_round_trips_and_honours_edits(tmp_path: Path):
    path = tmp_path / "out.csv"
    backfill.write_csv(
        path,
        [
            ("keep", "n", "d", "Find leads in seconds"),
            ("formula", "n", "d", "=not a formula once applied"),
            ("edited", "n", "d", "Generated line"),
            ("blank", "n", "d", "Will be blanked"),
            ("long", "n", "d", "Will be lengthened"),
        ],
    )
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    rows[3][3] = "Reviewer's line"
    rows[4][3] = "   "
    rows[5][3] = "x" * 101
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)

    assert backfill.read_reviewed_csv(path, 100) == [
        ("keep", "Find leads in seconds"),
        ("formula", "=not a formula once applied"),
        ("edited", "Reviewer's line"),
    ]


async def test_apply_persists_the_reviewed_csv_without_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "out.csv"
    backfill.write_csv(path, [("id-1", "n", "d", "Reviewer-approved line")])
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr("backend.data.db.connect", AsyncMock())
    monkeypatch.setattr("backend.data.db.disconnect", AsyncMock())
    generate = AsyncMock(side_effect=AssertionError("--apply must not generate"))
    monkeypatch.setattr(backfill, "generate_sub_headings", generate)
    monkeypatch.setattr(backfill, "find_listings_without_sub_heading", generate)
    persist = AsyncMock(return_value=1)
    monkeypatch.setattr(backfill, "persist_sub_headings", persist)
    before = path.read_bytes()

    assert await backfill.main(None, False, path, apply=True) == 0

    persist.assert_awaited_once_with([("id-1", "Reviewer-approved line")])
    assert path.read_bytes() == before


async def test_apply_without_a_reviewed_csv_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    connect = AsyncMock()
    monkeypatch.setattr("backend.data.db.connect", connect)

    with pytest.raises(SystemExit, match="run without --apply first"):
        await backfill.main(None, False, tmp_path / "missing.csv", apply=True)
    connect.assert_not_awaited()
