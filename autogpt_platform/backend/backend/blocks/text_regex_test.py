"""Regression tests for #15257 (Extract Text Information error handling)
and #15258 (Match Text Pattern forwarding falsy data)."""

from unittest.mock import MagicMock

import pytest

from backend.blocks.text import ExtractTextInformationBlock, MatchTextPatternBlock

NO_MATCH = [
    ("negative", "Hello, World!"),
    ("matched_results", []),
    ("matched_count", 0),
]


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


@pytest.mark.asyncio
@pytest.mark.parametrize("find_all", [True, False])
async def test_extract_skips_a_group_that_sat_out_a_match(find_all: bool):
    block = ExtractTextInformationBlock()
    input_data = block.Input(text="a b", pattern="(a)|(b)", group=2, find_all=find_all)

    assert await _outputs(block, input_data) == [
        ("positive", "b"),
        ("matched_results", ["b"]),
        ("matched_count", 1),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["(", "[a-", "(a+)+("])
async def test_extract_raises_on_an_invalid_pattern(pattern: str):
    block = ExtractTextInformationBlock()

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await _outputs(block, block.Input(text="abc", pattern=pattern))


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [2, -1])
async def test_extract_reports_no_match_for_a_group_the_pattern_lacks(group: int):
    block = ExtractTextInformationBlock()
    input_data = block.Input(text="Hello, World!", pattern="Hello, (.+)", group=group)

    assert await _outputs(block, input_data) == NO_MATCH


@pytest.mark.asyncio
async def test_extract_drops_matches_found_before_a_timeout(mocker):
    block = ExtractTextInformationBlock()

    def matches():
        match = MagicMock()
        match.groups.return_value = ()
        match.group.return_value = "Hello"
        yield match
        raise TimeoutError

    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.finditer.return_value = matches()
    input_data = block.Input(text="Hello, World!", pattern="Hello", find_all=True)

    assert await _outputs(block, input_data) == NO_MATCH


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [0, False, "", [], {}])
async def test_match_forwards_falsy_data(data):
    block = MatchTextPatternBlock()

    positive = await _outputs(block, block.Input(text="hello", match="hell", data=data))
    negative = await _outputs(block, block.Input(text="hello", match="xyz", data=data))

    assert positive == [("positive", data)]
    assert negative == [("negative", data)]
    # False == 0, so the values alone can't tell them apart
    assert type(positive[0][1]) is type(data)
    assert type(negative[0][1]) is type(data)


@pytest.mark.asyncio
async def test_match_forwards_the_text_when_data_is_none():
    block = MatchTextPatternBlock()

    positive = await _outputs(block, block.Input(text="hello", match="hell", data=None))
    negative = await _outputs(block, block.Input(text="hello", match="xyz", data=None))

    assert positive == [("positive", "hello")]
    assert negative == [("negative", "hello")]
