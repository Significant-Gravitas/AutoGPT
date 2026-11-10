"""Sheets API request builders for Google Sheets Set Layout."""

import re

from backend.blocks._base import BlockSchemaInput
from backend.data.model import SchemaField

from .sheets import _column_letter_to_index

# Sheets stops at column ZZZ, so a longer token can't be a real column.
_RANGE_PATTERN = re.compile(r"([A-Z]{1,3})(\d*):([A-Z]{1,3})(\d*)")
MAX_COLUMN_WIDTHS = 200
MAX_COLUMN_WIDTH_PX = 2000


class SheetColor(BlockSchemaInput):
    red: float = SchemaField(description="Red, 0 to 1", ge=0, le=1)
    green: float = SchemaField(description="Green, 0 to 1", ge=0, le=1)
    blue: float = SchemaField(description="Blue, 0 to 1", ge=0, le=1)


def split_sheet_prefix(a1_range: str) -> tuple[str | None, str]:
    """``"'My Leads'!A1:B2"`` -> ``("My Leads", "A1:B2")``; no prefix -> None."""
    if "!" not in a1_range:
        return None, a1_range
    sheet, cells = a1_range.rsplit("!", 1)
    sheet = sheet.strip()
    if len(sheet) >= 2 and sheet[0] == sheet[-1] == "'":
        sheet = sheet[1:-1].replace("''", "'")
    return sheet, cells


def grid_range(sheet_id: int, a1_range: str) -> dict:
    """A1 range ("A1:L200", "A:L" or "Sheet1!A2:C9") to a Sheets GridRange.

    The sheet prefix is not read here: the caller picks the target sheet.
    """
    _, cells = split_sheet_prefix(a1_range)
    match = _RANGE_PATTERN.fullmatch(cells.strip().replace("$", "").upper())
    if not match:
        raise ValueError(
            f"Invalid range '{a1_range}': use A1 notation like 'A1:L200' or 'A:L'"
        )
    start_col, start_row, end_col, end_row = match.groups()
    if bool(start_row) != bool(end_row):
        raise ValueError(
            f"Invalid range '{a1_range}': give a row on both ends or on neither"
        )
    grid: dict = {
        "sheetId": sheet_id,
        "startColumnIndex": _column_letter_to_index(start_col),
        "endColumnIndex": _column_letter_to_index(end_col) + 1,
    }
    if start_row:
        grid["startRowIndex"] = int(start_row) - 1
        grid["endRowIndex"] = int(end_row)
    return grid


def target_sheet_name(sheet_name: str, ranges: list[str]) -> str | None:
    """The sheet the block works on: ``sheet_name``, else the one sheet the
    ranges name, else None for the first sheet. A range naming a different
    sheet is an error, never a silent write to another tab."""
    prefixes = {p for p in (split_sheet_prefix(r)[0] for r in ranges if r) if p}
    target = sheet_name or None
    if target is None and len(prefixes) == 1:
        target = prefixes.pop()
    elif prefixes - {target}:
        named = ", ".join(sorted(prefixes))
        raise ValueError(
            f"The ranges name sheet(s) {named}, but the layout is for "
            f"'{target or 'the first sheet'}'. Set sheet_name to match."
        )
    return target


def freeze_requests(sheet_id: int, frozen_rows: int, frozen_columns: int) -> list:
    grid_properties: dict = {}
    if frozen_rows >= 0:
        grid_properties["frozenRowCount"] = frozen_rows
    if frozen_columns >= 0:
        grid_properties["frozenColumnCount"] = frozen_columns
    if not grid_properties:
        return []
    fields = ",".join(f"gridProperties.{key}" for key in grid_properties)
    return [
        {
            "updateSheetProperties": {
                "properties": {"sheetId": sheet_id, "gridProperties": grid_properties},
                "fields": fields,
            }
        }
    ]


def width_requests(sheet_id: int, column_widths: list[int]) -> list:
    if len(column_widths) > MAX_COLUMN_WIDTHS:
        raise ValueError(f"Give at most {MAX_COLUMN_WIDTHS} column widths")
    requests = []
    for index, width in enumerate(column_widths):
        if not 0 <= width <= MAX_COLUMN_WIDTH_PX:
            raise ValueError(
                f"Column width {width} is out of range: use 1 to "
                f"{MAX_COLUMN_WIDTH_PX} pixels, or 0 to leave a column as is"
            )
        if width == 0:
            continue
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": index,
                        "endIndex": index + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            }
        )
    return requests


def wrap_requests(sheet_id: int, wrap_text_range: str) -> list:
    if not wrap_text_range:
        return []
    return [
        {
            "repeatCell": {
                "range": grid_range(sheet_id, wrap_text_range),
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "TOP",
                    }
                },
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)",
            }
        }
    ]


def filter_requests(sheet_id: int, filter_range: str) -> list:
    if not filter_range:
        return []
    return [
        {"setBasicFilter": {"filter": {"range": grid_range(sheet_id, filter_range)}}}
    ]


def _overlaps(a: dict, b: dict) -> bool:
    """Whether two GridRanges share a cell; a missing bound is open-ended."""
    for start, end in (
        ("startRowIndex", "endRowIndex"),
        ("startColumnIndex", "endColumnIndex"),
    ):
        a_start, b_start = a.get(start, 0), b.get(start, 0)
        a_end, b_end = a.get(end, float("inf")), b.get(end, float("inf"))
        if a_start >= b_end or b_start >= a_end:
            return False
    return True


def banding_requests(
    sheet_id: int,
    banding_range: str,
    header_color: dict,
    band_color: dict,
    existing_bandings: list[dict],
) -> list:
    """Band a range, first removing only the bandings it overlaps: a cell can
    carry one banding, and bandings elsewhere on the sheet stay."""
    if not banding_range:
        return []
    target = grid_range(sheet_id, banding_range)
    requests: list = [
        {"deleteBanding": {"bandedRangeId": banding["bandedRangeId"]}}
        for banding in existing_bandings
        if _overlaps(target, banding.get("range", {}))
    ]
    requests.append(
        {
            "addBanding": {
                "bandedRange": {
                    "range": target,
                    "rowProperties": {
                        "headerColor": header_color,
                        "firstBandColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "secondBandColor": band_color,
                    },
                }
            }
        }
    )
    return requests


def layout_requests(
    sheet_id: int,
    *,
    frozen_rows: int,
    frozen_columns: int,
    column_widths: list[int],
    wrap_text_range: str,
    filter_range: str,
    banding_range: str,
    header_color: dict,
    band_color: dict,
    existing_bandings: list[dict],
) -> list[dict]:
    """The batchUpdate requests for a layout; an unset option sends nothing."""
    return [
        *freeze_requests(sheet_id, frozen_rows, frozen_columns),
        *width_requests(sheet_id, column_widths),
        *wrap_requests(sheet_id, wrap_text_range),
        *filter_requests(sheet_id, filter_range),
        *banding_requests(
            sheet_id, banding_range, header_color, band_color, existing_bandings
        ),
    ]
