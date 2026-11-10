"""Request building for GoogleSheetsSetLayoutBlock."""

from unittest.mock import MagicMock

import pytest

from backend.blocks.google._sheets_layout import (
    MAX_COLUMN_WIDTHS,
    SheetColor,
    grid_range,
    layout_requests,
    target_sheet_name,
)
from backend.blocks.google.sheets_layout import GoogleSheetsSetLayoutBlock

_UNSET = {
    "frozen_rows": -1,
    "frozen_columns": -1,
    "column_widths": [],
    "wrap_text_range": "",
    "filter_range": "",
    "banding_range": "",
    "header_color": {"red": 0.1, "green": 0.2, "blue": 0.3},
    "band_color": {"red": 0.9, "green": 0.9, "blue": 0.9},
    "existing_bandings": [],
}


def test_grid_range_with_rows_and_wide_columns():
    assert grid_range(7, "Leads!B2:AB10") == {
        "sheetId": 7,
        "startColumnIndex": 1,
        "endColumnIndex": 28,
        "startRowIndex": 1,
        "endRowIndex": 10,
    }


def test_grid_range_whole_columns():
    assert grid_range(0, "a:l") == {
        "sheetId": 0,
        "startColumnIndex": 0,
        "endColumnIndex": 12,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "A1",
        "A1:B",
        "1:2",
        "A1-B2",
        "",
        "AAAA1:B2",
        "A0:B0",
        "A5:B1",
        "C1:A5",
        "A1:B123456789",
        pytest.param("A" * 100_000 + "1:B2", id="oversized-column"),
    ],
)
def test_grid_range_rejects_bad_ranges(bad):
    with pytest.raises(ValueError):
        grid_range(0, bad)


def test_bad_range_error_does_not_echo_huge_input():
    with pytest.raises(ValueError) as raised:
        grid_range(0, "A" * 100_000 + "1:B2")
    assert len(str(raised.value)) < 200


def test_target_sheet_comes_from_the_ranges_when_not_named():
    assert target_sheet_name("", ["Leads!A1:P1000", "", "'Leads'!A2:P9"]) == "Leads"
    assert target_sheet_name("", ["A1:B2"]) is None
    assert target_sheet_name("Leads", ["Leads!A1:B2", "A1:B2"]) == "Leads"
    assert target_sheet_name("", ["'Bob''s Leads'!A1:B2"]) == "Bob's Leads"


@pytest.mark.parametrize(
    "sheet_name, ranges",
    [("Leads", ["Other!A1:B2"]), ("", ["Leads!A1:B2", "Other!A1:B2"])],
)
def test_a_range_on_another_sheet_is_refused(sheet_name, ranges):
    with pytest.raises(ValueError, match="Set sheet_name"):
        target_sheet_name(sheet_name, ranges)


def test_nothing_set_sends_nothing():
    assert layout_requests(3, **_UNSET) == []


def test_freeze_only_names_the_fields_it_sets():
    [request] = layout_requests(3, **{**_UNSET, "frozen_rows": 1})
    update = request["updateSheetProperties"]
    assert update["properties"] == {
        "sheetId": 3,
        "gridProperties": {"frozenRowCount": 1},
    }
    assert update["fields"] == "gridProperties.frozenRowCount"


def test_freeze_rows_and_columns_together():
    [request] = layout_requests(3, **{**_UNSET, "frozen_rows": 1, "frozen_columns": 2})
    assert (
        request["updateSheetProperties"]["fields"]
        == "gridProperties.frozenRowCount,gridProperties.frozenColumnCount"
    )


def test_zero_freeze_unfreezes():
    [request] = layout_requests(3, **{**_UNSET, "frozen_columns": 0})
    assert request["updateSheetProperties"]["properties"]["gridProperties"] == {
        "frozenColumnCount": 0
    }


def test_column_widths_skip_zero():
    requests = layout_requests(3, **{**_UNSET, "column_widths": [200, 0, 90]})
    sized = [
        (
            r["updateDimensionProperties"]["range"]["startIndex"],
            r["updateDimensionProperties"]["properties"]["pixelSize"],
        )
        for r in requests
    ]
    assert sized == [(0, 200), (2, 90)]


@pytest.mark.parametrize("widths", [[10**12], [-5], [100] * (MAX_COLUMN_WIDTHS + 1)])
def test_column_widths_are_bounded(widths):
    with pytest.raises(ValueError):
        layout_requests(3, **{**_UNSET, "column_widths": widths})


def test_wrap_and_filter():
    wrap, flt = layout_requests(
        3, **{**_UNSET, "wrap_text_range": "A2:C50", "filter_range": "A1:C50"}
    )
    assert wrap["repeatCell"]["cell"]["userEnteredFormat"] == {
        "wrapStrategy": "WRAP",
        "verticalAlignment": "TOP",
    }
    assert (
        wrap["repeatCell"]["fields"]
        == "userEnteredFormat(wrapStrategy,verticalAlignment)"
    )
    assert flt["setBasicFilter"]["filter"]["range"]["endColumnIndex"] == 3


def test_banding_replaces_only_overlapping_banding():
    existing = [
        {
            "bandedRangeId": 11,
            "range": {
                "sheetId": 3,
                "startRowIndex": 0,
                "endRowIndex": 20,
                "startColumnIndex": 0,
                "endColumnIndex": 2,
            },
        },
        {
            "bandedRangeId": 12,
            "range": {
                "sheetId": 3,
                "startRowIndex": 0,
                "endRowIndex": 20,
                "startColumnIndex": 10,
                "endColumnIndex": 12,
            },
        },
        {
            "bandedRangeId": 13,
            "range": {"sheetId": 3, "startColumnIndex": 1, "endColumnIndex": 2},
        },
    ]
    requests = layout_requests(
        3, **{**_UNSET, "banding_range": "A1:C50", "existing_bandings": existing}
    )
    assert requests[:2] == [
        {"deleteBanding": {"bandedRangeId": 11}},
        {"deleteBanding": {"bandedRangeId": 13}},
    ]
    props = requests[2]["addBanding"]["bandedRange"]["rowProperties"]
    assert props["headerColor"] == _UNSET["header_color"]
    assert props["secondBandColor"] == _UNSET["band_color"]


def test_banding_that_only_touches_the_range_is_kept():
    # A1:C50 covers columns 0-2 and rows 0-49; these start exactly where it ends.
    existing = [
        {
            "bandedRangeId": 21,
            "range": {"sheetId": 3, "startColumnIndex": 3, "endColumnIndex": 5},
        },
        {
            "bandedRangeId": 22,
            "range": {"sheetId": 3, "startRowIndex": 50, "endRowIndex": 60},
        },
    ]
    requests = layout_requests(
        3, **{**_UNSET, "banding_range": "A1:C50", "existing_bandings": existing}
    )
    assert [next(iter(r)) for r in requests] == ["addBanding"]


def test_existing_banding_kept_when_banding_not_asked():
    existing = [{"bandedRangeId": 11, "range": {"sheetId": 3}}]
    assert layout_requests(3, **{**_UNSET, "existing_bandings": existing}) == []


@pytest.mark.parametrize(
    "bad", [{"red": "x", "green": 0, "blue": 0}, {"red": 2, "green": 0, "blue": 0}]
)
def test_colors_are_typed_and_bounded(bad):
    with pytest.raises(ValueError):
        SheetColor(**bad)


def _service(sheets: list[dict]) -> MagicMock:
    service = MagicMock()
    service.spreadsheets().get().execute.return_value = {"sheets": sheets}
    return service


def _input(**kwargs):
    return GoogleSheetsSetLayoutBlock.Input(
        spreadsheet={
            "id": "s1",
            "name": "Leads",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        **kwargs,
    )


def test_set_layout_targets_the_sheet_the_range_names_with_one_read():
    service = _service(
        [
            {"properties": {"sheetId": 0, "title": "Sheet1"}},
            {"properties": {"sheetId": 9, "title": "Leads"}, "bandedRanges": []},
        ]
    )
    result = GoogleSheetsSetLayoutBlock()._set_layout(
        service, "s1", _input(banding_range="Leads!A1:P1000")
    )
    assert result == {"success": True, "requests": 1}
    body = service.spreadsheets().batchUpdate.call_args.kwargs["body"]
    added = body["requests"][0]["addBanding"]["bandedRange"]
    assert added["range"]["sheetId"] == 9
    assert added["rowProperties"]["headerColor"] == {
        "red": 0.13,
        "green": 0.2,
        "blue": 0.33,
    }
    # one metadata read, not two
    assert service.spreadsheets().get().execute.call_count == 1


def test_set_layout_checks_inputs_before_reading_the_sheet():
    service = _service([{"properties": {"sheetId": 0, "title": "Sheet1"}}])
    block = GoogleSheetsSetLayoutBlock()
    with pytest.raises(ValueError, match="Nothing to change"):
        block._set_layout(service, "s1", _input())
    with pytest.raises(ValueError, match="Invalid range"):
        block._set_layout(service, "s1", _input(filter_range="A0:B0"))
    assert service.spreadsheets().get().execute.call_count == 0


def test_set_layout_refuses_a_missing_sheet():
    service = _service([{"properties": {"sheetId": 0, "title": "Sheet1"}}])
    with pytest.raises(ValueError, match="not found"):
        GoogleSheetsSetLayoutBlock()._set_layout(
            service, "s1", _input(sheet_name="Leads", frozen_rows=1)
        )
