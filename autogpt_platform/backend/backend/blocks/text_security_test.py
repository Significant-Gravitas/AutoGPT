from unittest.mock import MagicMock

import pytest

from backend.blocks.text import (
    MAX_REGEX_PATTERN_LENGTH,
    MAX_REGEX_TEXT_LENGTH,
    REGEX_TIMEOUT_SECONDS,
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
