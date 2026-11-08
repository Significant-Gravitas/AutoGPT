"""Regression tests for #15257 (ExtractTextInformationBlock error handling)
and #15258 (MatchTextPatternBlock forwarding falsy data)."""

import pytest

from backend.blocks.text import ExtractTextInformationBlock, MatchTextPatternBlock
from backend.util.user_regex import MAX_COMPILED_ITEMS, MAX_PATTERN_LENGTH

FALSY_DATA = [0, False, "", [], {}]


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


@pytest.mark.asyncio
@pytest.mark.parametrize("find_all", [True, False])
async def test_extract_skips_unmatched_optional_group(find_all: bool):
    block = ExtractTextInformationBlock()
    out = await _outputs(
        block,
        block.Input(text="a b", pattern="(a)|(b)", group=2, find_all=find_all),
    )
    assert out == [
        ("positive", "b"),
        ("matched_results", ["b"]),
        ("matched_count", 1),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["(", "[a-", "(a+)+("])
async def test_extract_rejects_invalid_pattern(pattern: str):
    block = ExtractTextInformationBlock()
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await _outputs(block, block.Input(text="abc", pattern=pattern))


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [5, -1])
async def test_extract_rejects_invalid_group_number(group: int):
    block = ExtractTextInformationBlock()
    with pytest.raises(ValueError, match="Invalid group number"):
        await _outputs(
            block, block.Input(text="Hello, World!", pattern="Hello, (.+)", group=group)
        )


@pytest.mark.asyncio
async def test_extract_timeout_still_returns_empty_results(mocker):
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.groups = 0
    compile_pattern.return_value.finditer.side_effect = TimeoutError
    block = ExtractTextInformationBlock()
    out = await _outputs(block, block.Input(text="order 42", pattern=r"\d\d"))
    assert out == [
        ("negative", "order 42"),
        ("matched_results", []),
        ("matched_count", 0),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern, reason",
    [
        ("a" * (MAX_PATTERN_LENGTH + 1), "too large"),
        (f"a{{{MAX_COMPILED_ITEMS}}}", "repeats too much"),
    ],
)
async def test_extract_oversized_pattern_is_an_error(pattern: str, reason: str):
    block = ExtractTextInformationBlock()
    with pytest.raises(ValueError, match=reason):
        await _outputs(block, block.Input(text="order 42", pattern=pattern))


@pytest.mark.asyncio
@pytest.mark.parametrize("data", FALSY_DATA)
async def test_match_forwards_falsy_data_on_positive_and_negative(data):
    block = MatchTextPatternBlock()
    positive = await _outputs(block, block.Input(text="hello", match="hell", data=data))
    negative = await _outputs(block, block.Input(text="hello", match="xyz", data=data))
    assert positive == [("positive", data)]
    assert negative == [("negative", data)]


@pytest.mark.asyncio
async def test_match_falls_back_to_text_when_data_is_none():
    block = MatchTextPatternBlock()
    positive = await _outputs(block, block.Input(text="hello", match="hell", data=None))
    negative = await _outputs(block, block.Input(text="hello", match="xyz", data=None))
    assert positive == [("positive", "hello")]
    assert negative == [("negative", "hello")]


@pytest.mark.asyncio
async def test_match_falls_back_to_text_when_data_unconnected():
    block = MatchTextPatternBlock()
    positive = await _outputs(block, block.Input(text="hello", match="hell"))
    negative = await _outputs(block, block.Input(text="hello", match="xyz"))
    assert positive == [("positive", "hello")]
    assert negative == [("negative", "hello")]
