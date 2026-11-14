"""Regression tests for FileReadBlock delimiter and size-limit chunking."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.text import FileReadBlock
from backend.data.execution import ExecutionContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delimiter", "skip_rows", "row_limit", "size_limit", "expected"),
    [
        ("", 0, 0, 0, ["one\ntwo\nthree"]),
        ("\n", 0, 0, 0, ["one", "two", "three"]),
        ("\n", 1, 1, 0, ["two"]),
        ("\n", 0, 0, 2, ["on", "e", "tw", "o", "th", "re", "e"]),
        ("", 0, 0, 4, ["one\n", "two\n", "thre", "e"]),
        ("\n", 5, 0, 0, [""]),
    ],
)
async def test_file_read_chunks_each_row_without_rejoining(
    tmp_path, delimiter, skip_rows, row_limit, size_limit, expected
):
    file_path = tmp_path / "rows.txt"
    file_path.write_text("one\ntwo\nthree", encoding="utf-8")

    block = FileReadBlock()
    input_data = FileReadBlock.Input(
        file_input="data:text/plain;base64,b25l",
        delimiter=delimiter,
        skip_rows=skip_rows,
        row_limit=row_limit,
        size_limit=size_limit,
    )
    context = ExecutionContext(user_id="test-user", graph_exec_id="test-exec")

    with (
        patch(
            "backend.blocks.text.store_media_file",
            new_callable=AsyncMock,
            return_value="rows.txt",
        ),
        patch(
            "backend.blocks.text.get_exec_file_path",
            return_value=str(file_path),
        ),
    ):
        actual = [
            item async for item in block.run(input_data, execution_context=context)
        ]

    assert actual == [("content", chunk) for chunk in expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("size_limit", [0, 3])
async def test_file_read_emits_empty_content_for_blank_rows(tmp_path, size_limit):
    file_path = tmp_path / "rows.txt"
    file_path.write_text("one\n\nthree", encoding="utf-8")

    block = FileReadBlock()
    input_data = FileReadBlock.Input(
        file_input="data:text/plain;base64,b25l",
        delimiter="\n",
        size_limit=size_limit,
    )
    context = ExecutionContext(user_id="test-user", graph_exec_id="test-exec")

    with (
        patch(
            "backend.blocks.text.store_media_file",
            new_callable=AsyncMock,
            return_value="rows.txt",
        ),
        patch(
            "backend.blocks.text.get_exec_file_path",
            return_value=str(file_path),
        ),
    ):
        actual = [
            item async for item in block.run(input_data, execution_context=context)
        ]

    expected = ["one", "", "three"] if size_limit == 0 else ["one", "", "thr", "ee"]
    assert actual == [("content", chunk) for chunk in expected]
