from unittest.mock import MagicMock

import pytest

from backend.blocks.text import (
    MAX_MATCHES,
    MAX_REGEX_PATTERN_LENGTH,
    MAX_REGEX_TEXT_LENGTH,
    REGEX_TIMEOUT_SECONDS,
    ExtractTextInformationBlock,
    MatchTextPatternBlock,
)


async def _collect_outputs(block, input_data):
    return [item async for item in block.run(input_data)]


@pytest.mark.asyncio
async def test_match_text_pattern_uses_timeout_for_every_pattern(mocker):
    block = MatchTextPatternBlock()
    search = mocker.patch(
        "backend.blocks.text.regex.search",
        return_value=MagicMock(),
    )
    input_data = block.Input(
        text="ordinary text",
        match="ordinary",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    outputs = await _collect_outputs(block, input_data)

    assert outputs == [("positive", "matched")]
    search.assert_called_once_with(
        "ordinary",
        "ordinary text",
        flags=mocker.ANY,
        timeout=REGEX_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
async def test_match_text_pattern_surfaces_regex_timeout_as_input_error(mocker):
    block = MatchTextPatternBlock()
    mocker.patch(
        "backend.blocks.text.regex.search",
        side_effect=TimeoutError,
    )
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
async def test_match_text_pattern_rejects_oversized_pattern_before_search(mocker):
    block = MatchTextPatternBlock()
    search = mocker.patch("backend.blocks.text.regex.search")
    input_data = block.Input(
        text="ordinary text",
        match="a" * (MAX_REGEX_PATTERN_LENGTH + 1),
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match="pattern is too large"):
        await _collect_outputs(block, input_data)

    search.assert_not_called()


@pytest.mark.asyncio
async def test_match_text_pattern_rejects_oversized_text_before_search(mocker):
    block = MatchTextPatternBlock()
    search = mocker.patch("backend.blocks.text.regex.search")
    input_data = block.Input(
        text="a" * (MAX_REGEX_TEXT_LENGTH + 1),
        match="a",
        data="matched",
        case_sensitive=True,
        dot_all=True,
    )

    with pytest.raises(ValueError, match="too large"):
        await _collect_outputs(block, input_data)

    search.assert_not_called()


NO_MATCH_OUTPUTS = [
    ("negative", "order 42"),
    ("matched_results", []),
    ("matched_count", 0),
]


@pytest.mark.asyncio
async def test_extract_text_information_uses_timeout_for_every_pattern(mocker):
    block = ExtractTextInformationBlock()
    finditer = mocker.patch(
        "backend.blocks.text.regex.finditer",
        return_value=iter([]),
    )
    input_data = block.Input(text="order 42", pattern=r"\d\d")

    await _collect_outputs(block, input_data)

    finditer.assert_called_once_with(
        r"\d\d",
        "order 42",
        flags=mocker.ANY,
        timeout=REGEX_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
async def test_extract_text_information_returns_no_matches_when_the_regex_times_out(
    mocker,
):
    block = ExtractTextInformationBlock()
    mocker.patch(
        "backend.blocks.text.regex.finditer",
        side_effect=TimeoutError,
    )
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

    mocker.patch("backend.blocks.text.regex.finditer", return_value=matches())
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

    mocker.patch("backend.blocks.text.regex.finditer", return_value=matches())
    input_data = block.Input(text="numbers", pattern=r"\d+", find_all=True)

    outputs = await _collect_outputs(block, input_data)

    assert ("matched_count", MAX_MATCHES) in outputs
    assert pulled == MAX_MATCHES


@pytest.mark.asyncio
async def test_extract_text_information_skips_an_oversized_pattern(mocker):
    block = ExtractTextInformationBlock()
    finditer = mocker.patch("backend.blocks.text.regex.finditer")
    # Matches "order" and is valid, but too long to compile safely
    pattern = "order" + "(?:)" * (MAX_REGEX_PATTERN_LENGTH // 4 + 1)
    input_data = block.Input(text="order 42", pattern=pattern)

    assert await _collect_outputs(block, input_data) == NO_MATCH_OUTPUTS
    finditer.assert_not_called()
