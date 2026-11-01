import pytest

from backend.blocks.text import (
    MAX_REGEX_PATTERN_LENGTH,
    MAX_REGEX_REPEAT,
    ExtractTextInformationBlock,
    MatchTextPatternBlock,
)
from backend.util.test import execute_block_test

LIMIT_MESSAGE = f"maximum {MAX_REGEX_REPEAT} repetitions"

NO_MATCH_OUTPUTS = [
    ("negative", "order 42"),
    ("matched_results", []),
    ("matched_count", 0),
]


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


def _match_input(text: str, pattern: str) -> MatchTextPatternBlock.Input:
    return MatchTextPatternBlock.Input(text=text, match=pattern, data="matched")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [
        "a{30000000}",
        "a{2,30000000}",
        "x{2}y{30000000}",
        f"a{{{MAX_REGEX_REPEAT + 1}}}",
    ],
)
async def test_match_text_pattern_refuses_large_repeat_before_compiling(
    mocker, pattern
):
    block = MatchTextPatternBlock()
    search = mocker.patch("backend.blocks.text.regex.search")

    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        await _outputs(block, _match_input("aaa", pattern))

    search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["(a{100}){101}", "((a{10}){10}){101}"])
async def test_match_text_pattern_refuses_large_nested_repeat_before_compiling(
    mocker, pattern
):
    block = MatchTextPatternBlock()
    search = mocker.patch("backend.blocks.text.regex.search")

    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        await _outputs(block, _match_input("aaa", pattern))

    search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [f"a{{{MAX_REGEX_REPEAT}}}", "((a{10}){10}){100}", f"a{{{MAX_REGEX_REPEAT},}}"],
)
async def test_match_text_pattern_allows_repeat_at_the_limit(pattern):
    block = MatchTextPatternBlock()
    text = "a" * MAX_REGEX_REPEAT

    assert await _outputs(block, _match_input(text, pattern)) == [
        ("positive", "matched")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [r"x\{30000000\}", r"x[{]30000000}"])
async def test_match_text_pattern_treats_literal_braces_as_text(pattern):
    block = MatchTextPatternBlock()

    assert await _outputs(block, _match_input("x{30000000}", pattern)) == [
        ("positive", "matched")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, pattern, expected",
    [
        ("call 555-1234", r"\d{3}-\d{4}", "positive"),
        ("call 55-1234", r"\d{3}-\d{4}", "negative"),
        ("héllo", r"\p{L}{5}", "positive"),
        ("abab", "(ab){2}", "positive"),
    ],
)
async def test_match_text_pattern_ordinary_repeats_unchanged(text, pattern, expected):
    block = MatchTextPatternBlock()

    assert await _outputs(block, _match_input(text, pattern)) == [(expected, "matched")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern", ["a{30000000}", "(a{100}){101}", f"a{{{MAX_REGEX_REPEAT + 1}}}"]
)
async def test_extract_text_information_refuses_large_repeat_before_compiling(
    mocker, pattern
):
    block = ExtractTextInformationBlock()
    finditer = mocker.patch("backend.blocks.text.regex.finditer")
    input_data = block.Input(text="order 42", pattern=pattern)
    outputs = []

    with pytest.raises(ValueError, match=LIMIT_MESSAGE):
        async for output in block.run(input_data):
            outputs.append(output)

    assert outputs == []
    finditer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["a" * (MAX_REGEX_PATTERN_LENGTH + 1), "(unclosed"])
async def test_extract_text_information_other_pattern_errors_still_give_empty_results(
    pattern,
):
    block = ExtractTextInformationBlock()
    input_data = block.Input(text="order 42", pattern=pattern)

    assert await _outputs(block, input_data) == NO_MATCH_OUTPUTS


@pytest.mark.asyncio
async def test_extract_text_information_ordinary_repeats_unchanged():
    block = ExtractTextInformationBlock()
    input_data = block.Input(
        text="orders 42 and 1234", pattern=r"\d{2,4}", find_all=True
    )

    assert await _outputs(block, input_data) == [
        ("positive", "42"),
        ("positive", "1234"),
        ("matched_results", ["42", "1234"]),
        ("matched_count", 2),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_class", [MatchTextPatternBlock, ExtractTextInformationBlock]
)
async def test_regex_blocks_keep_their_documented_outputs(block_class):
    await execute_block_test(block_class())
