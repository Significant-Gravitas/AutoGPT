from unittest.mock import MagicMock

import pytest

from backend.blocks.text import (
    MAX_MATCHES,
    MAX_REGEX_TEXT_LENGTH,
    REGEX_TIMEOUT_SECONDS,
    ExtractTextInformationBlock,
    MatchTextPatternBlock,
)
from backend.util.user_regex import MAX_COMPILED_ITEMS, MAX_PATTERN_LENGTH

# Refused before compiling: one too long, one that repeats past the limit.
OVERSIZED_PATTERNS = [
    ("a" * (MAX_PATTERN_LENGTH + 1), "too large"),
    (f"a{{{MAX_COMPILED_ITEMS}}}", "repeats too much"),
]


async def _collect_outputs(block, input_data):
    return [item async for item in block.run(input_data)]


@pytest.mark.asyncio
async def test_match_text_pattern_uses_timeout_for_every_pattern(mocker):
    block = MatchTextPatternBlock()
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    input_data = block.Input(
        text="ordinary text",
        match="ordinary",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    outputs = await _collect_outputs(block, input_data)

    assert outputs == [("positive", "matched")]
    compile_pattern.assert_called_once_with("ordinary", mocker.ANY)
    compile_pattern.return_value.search.assert_called_once_with(
        "ordinary text", timeout=REGEX_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_match_text_pattern_surfaces_regex_timeout_as_input_error(mocker):
    block = MatchTextPatternBlock()
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.search.side_effect = TimeoutError
    input_data = block.Input(
        text="a" * 40 + "!",
        match=r"(a+)+$",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match="timed out"):
        await _collect_outputs(block, input_data)


@pytest.mark.asyncio
async def test_match_text_pattern_times_out_catastrophic_regex(mocker):
    mocker.patch("backend.blocks.text.REGEX_TIMEOUT_SECONDS", 0.01)
    block = MatchTextPatternBlock()
    input_data = block.Input(
        text="a" * 1_000 + "!",
        match=r"^(a|aa)+$",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match="timed out"):
        await _collect_outputs(block, input_data)


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern, error", OVERSIZED_PATTERNS)
async def test_match_text_pattern_rejects_oversized_pattern_before_compiling(
    mocker, pattern: str, error: str
):
    block = MatchTextPatternBlock()
    compile_pattern = mocker.patch("backend.util.user_regex.regex.compile")
    input_data = block.Input(
        text="ordinary text",
        match=pattern,
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match=error):
        await _collect_outputs(block, input_data)

    compile_pattern.assert_not_called()


@pytest.mark.asyncio
async def test_match_text_pattern_rejects_oversized_text_before_search(mocker):
    block = MatchTextPatternBlock()
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    input_data = block.Input(
        text="a" * (MAX_REGEX_TEXT_LENGTH + 1),
        match="a",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match="too large"):
        await _collect_outputs(block, input_data)

    compile_pattern.assert_not_called()


NO_MATCH_OUTPUTS = [
    ("negative", "order 42"),
    ("matched_results", []),
    ("matched_count", 0),
]


@pytest.mark.asyncio
async def test_extract_text_information_uses_timeout_for_every_pattern(mocker):
    block = ExtractTextInformationBlock()
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.finditer.return_value = iter([])
    input_data = block.Input(text="order 42", pattern=r"\d\d")

    await _collect_outputs(block, input_data)

    compile_pattern.assert_called_once_with(r"\d\d", mocker.ANY)
    compile_pattern.return_value.finditer.assert_called_once_with(
        "order 42", timeout=REGEX_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_extract_text_information_returns_no_matches_when_the_regex_times_out(
    mocker,
):
    block = ExtractTextInformationBlock()
    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.finditer.side_effect = TimeoutError
    input_data = block.Input(text="order 42", pattern=r"\d\d")

    assert await _collect_outputs(block, input_data) == NO_MATCH_OUTPUTS


@pytest.mark.asyncio
async def test_extract_text_information_stops_at_the_first_match_unless_find_all(
    mocker,
):
    block = ExtractTextInformationBlock()
    pulled: list[str] = []

    def matches():
        for found in ("12", "34"):
            pulled.append(found)
            match = MagicMock()
            match.groups.return_value = ()
            match.group.return_value = found
            yield match

    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.finditer.return_value = matches()
    input_data = block.Input(text="12 34", pattern=r"\d\d", find_all=False)

    outputs = await _collect_outputs(block, input_data)

    assert outputs[0] == ("positive", "12")
    assert pulled == ["12"]


@pytest.mark.asyncio
async def test_extract_text_information_stops_searching_at_the_match_cap(mocker):
    block = ExtractTextInformationBlock()
    pulled = 0

    def matches():
        nonlocal pulled
        for number in range(MAX_MATCHES + 1):
            pulled += 1
            match = MagicMock()
            match.groups.return_value = ()
            match.group.return_value = str(number)
            yield match

    compile_pattern = mocker.patch("backend.blocks.text.compile_user_regex")
    compile_pattern.return_value.finditer.return_value = matches()
    input_data = block.Input(text="numbers", pattern=r"\d+", find_all=True)

    outputs = await _collect_outputs(block, input_data)

    assert ("matched_count", MAX_MATCHES) in outputs
    assert pulled == MAX_MATCHES


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [pattern for pattern, _ in OVERSIZED_PATTERNS])
async def test_extract_text_information_skips_an_oversized_pattern(
    mocker, pattern: str
):
    block = ExtractTextInformationBlock()
    compile_pattern = mocker.patch("backend.util.user_regex.regex.compile")
    input_data = block.Input(text="order 42", pattern=pattern)

    assert await _collect_outputs(block, input_data) == NO_MATCH_OUTPUTS
    compile_pattern.assert_not_called()
