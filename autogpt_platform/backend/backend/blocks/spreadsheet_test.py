"""Tests for ReadSpreadsheetBlock Excel/CSV fidelity (#14638)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from backend.blocks.spreadsheet import ReadSpreadsheetBlock
from backend.data.execution import ExecutionContext
from backend.util.type import MediaFileType


def _ctx() -> ExecutionContext:
    return ExecutionContext(user_id="test-user", graph_exec_id="test-exec")


async def _collect(block, input_data, execution_context=None):
    ctx = execution_context or _ctx()
    outputs = []
    async for name, value in block.run(input_data, execution_context=ctx):
        outputs.append((name, value))
    return outputs


async def _run_excel(tmp_path: Path, workbook_name: str, **input_kwargs):
    """Run the block against a real .xlsx via mocked media-file plumbing."""
    xlsx_path = tmp_path / workbook_name
    block = ReadSpreadsheetBlock()
    input_data = block.Input(
        file_input=MediaFileType(workbook_name),
        **input_kwargs,
    )

    async def fake_store(*, file, execution_context, return_format):
        return workbook_name

    with (
        patch(
            "backend.blocks.spreadsheet.store_media_file",
            new=AsyncMock(side_effect=fake_store),
        ),
        patch(
            "backend.blocks.spreadsheet.get_exec_file_path",
            return_value=str(xlsx_path),
        ),
    ):
        return await _collect(block, input_data)


@pytest.mark.asyncio
async def test_built_in_csv_test_vectors():
    """Existing block test_input/test_output still pass."""
    block = ReadSpreadsheetBlock()
    outs = await _collect(
        block,
        block.Input(contents="a, b, c\n1,2,3\n4,5,6", produce_singular_result=False),
    )
    assert outs == [
        (
            "rows",
            [
                {"a": "1", "b": "2", "c": "3"},
                {"a": "4", "b": "5", "c": "6"},
            ],
        )
    ]
    outs = await _collect(
        block,
        block.Input(contents="a, b, c\n1,2,3\n4,5,6", produce_singular_result=True),
    )
    assert outs == [
        ("row", {"a": "1", "b": "2", "c": "3"}),
        ("row", {"a": "4", "b": "5", "c": "6"}),
    ]


@pytest.mark.asyncio
async def test_excel_preserves_literal_na_strings(tmp_path):
    rows = [
        ("A1", "N/A"),
        ("A2", "NULL"),
        ("A3", "NaN"),
        ("A4", "None"),
        ("A5", "nan"),
        ("A6", "n/a"),
        ("A7", "NA"),
        ("A8", "ok"),
    ]
    pd.DataFrame(rows, columns=["Code", "Status"]).to_excel(
        tmp_path / "status.xlsx", index=False
    )

    outputs = await _run_excel(tmp_path, "status.xlsx")
    assert outputs[0][0] == "rows"
    result = outputs[0][1]
    assert [r["Status"] for r in result] == [
        "N/A",
        "NULL",
        "NaN",
        "None",
        "nan",
        "n/a",
        "NA",
        "ok",
    ]


@pytest.mark.asyncio
async def test_excel_int_with_gap_stays_int_like_string(tmp_path):
    pd.DataFrame(
        [("Cable", 12, "spare"), ("Hub", None, "n/a")],
        columns=["Product", "Units", "Note"],
    ).to_excel(tmp_path / "units.xlsx", index=False)

    outputs = await _run_excel(tmp_path, "units.xlsx")
    result = outputs[0][1]
    assert result[0]["Units"] == "12"
    assert result[0]["Units"] != "12.0"
    assert result[1]["Units"] == ""
    assert result[1]["Note"] == "n/a"


@pytest.mark.asyncio
async def test_excel_defaults_to_first_sheet(tmp_path):
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Q1", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Q2", index=False)

    outputs = await _run_excel(tmp_path, "multi.xlsx")
    assert outputs[0][1] == [{"a": "1"}]


@pytest.mark.asyncio
async def test_excel_sheet_name_selects_named_sheet(tmp_path):
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Q1", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Q2", index=False)

    outputs = await _run_excel(tmp_path, "multi.xlsx", sheet_name="Q2")
    assert outputs[0][1] == [{"b": "2"}]


@pytest.mark.asyncio
async def test_skip_columns_by_header_name():
    block = ReadSpreadsheetBlock()
    input_data = block.Input(
        contents="a,b,c\n1,2,3\n4,5,6",
        skip_columns=["b"],
        produce_singular_result=False,
    )
    outputs = await _collect(block, input_data)
    assert outputs[0][1] == [
        {"a": "1", "c": "3"},
        {"a": "4", "c": "6"},
    ]


@pytest.mark.asyncio
async def test_skip_columns_by_index_without_header():
    block = ReadSpreadsheetBlock()
    input_data = block.Input(
        contents="1,2,3\n4,5,6",
        has_header=False,
        skip_columns=["1"],
        produce_singular_result=False,
    )
    outputs = await _collect(block, input_data)
    assert outputs[0][1] == [
        {"0": "1", "2": "3"},
        {"0": "4", "2": "6"},
    ]


@pytest.mark.asyncio
async def test_excel_respects_delimiter(tmp_path):
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(tmp_path / "delim.xlsx", index=False)
    outputs = await _run_excel(tmp_path, "delim.xlsx", delimiter=";")
    assert outputs[0][1] == [{"a": "1", "b": "2"}]
