"""Request building for GoogleSheetsSetLayoutBlock."""

import pytest

from backend.blocks.google.sheets import _grid_range, _layout_requests

_UNSET = dict(
    frozen_rows=-1,
    frozen_columns=-1,
    column_widths=[],
    wrap_text_range="",
    filter_range="",
    banding_range="",
    header_color={"red": 0.1, "green": 0.2, "blue": 0.3},
    band_color={"red": 0.9, "green": 0.9, "blue": 0.9},
    existing_banding_ids=[],
)


def test_grid_range_with_rows_and_wide_columns():
    assert _grid_range(7, "Leads!B2:AB10") == {
        "sheetId": 7,
        "startColumnIndex": 1,
        "endColumnIndex": 28,
        "startRowIndex": 1,
        "endRowIndex": 10,
    }


def test_grid_range_whole_columns():
    assert _grid_range(0, "a:l") == {
        "sheetId": 0,
        "startColumnIndex": 0,
        "endColumnIndex": 12,
    }


@pytest.mark.parametrize("bad", ["A1", "A1:B", "1:2", "A1-B2", ""])
def test_grid_range_rejects_bad_ranges(bad):
    with pytest.raises(ValueError):
        _grid_range(0, bad)


def test_nothing_set_sends_nothing():
    assert _layout_requests(3, **_UNSET) == []


def test_freeze_only_names_the_fields_it_sets():
    [request] = _layout_requests(3, **{**_UNSET, "frozen_rows": 1})
    update = request["updateSheetProperties"]
    assert update["properties"] == {
        "sheetId": 3,
        "gridProperties": {"frozenRowCount": 1},
    }
    assert update["fields"] == "gridProperties.frozenRowCount"


def test_zero_freeze_unfreezes():
    [request] = _layout_requests(3, **{**_UNSET, "frozen_columns": 0})
    assert request["updateSheetProperties"]["properties"]["gridProperties"] == {
        "frozenColumnCount": 0
    }


def test_column_widths_skip_zero():
    requests = _layout_requests(3, **{**_UNSET, "column_widths": [200, 0, 90]})
    sized = [
        (
            r["updateDimensionProperties"]["range"]["startIndex"],
            r["updateDimensionProperties"]["properties"]["pixelSize"],
        )
        for r in requests
    ]
    assert sized == [(0, 200), (2, 90)]


def test_wrap_and_filter():
    wrap, flt = _layout_requests(
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


def test_banding_replaces_existing_banding():
    requests = _layout_requests(
        3, **{**_UNSET, "banding_range": "A1:C50", "existing_banding_ids": [11, 12]}
    )
    assert requests[:2] == [
        {"deleteBanding": {"bandedRangeId": 11}},
        {"deleteBanding": {"bandedRangeId": 12}},
    ]
    props = requests[2]["addBanding"]["bandedRange"]["rowProperties"]
    assert props["headerColor"] == _UNSET["header_color"]
    assert props["secondBandColor"] == _UNSET["band_color"]


def test_existing_banding_kept_when_banding_not_asked():
    assert _layout_requests(3, **{**_UNSET, "existing_banding_ids": [11]}) == []
