"""Regression tests for #15257 (ExtractTextInformationBlock error handling)
and #15258 (MatchTextPatternBlock forwarding falsy data)."""

import pytest

from backend.blocks.text import ExtractTextInformationBlock, MatchTextPatternBlock


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


# --- #15257 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_skips_unmatched_optional_group():
    block = ExtractTextInformationBlock()
    out = await _outputs(
        block,
        block.Input(text="a b", pattern="(a)|(b)", group=2, find_all=True),
    )
    # Before the fix, group 2 is None on the "a" match → len(None) TypeError,
    # swallowed into "no match" and the "b" match is lost.
    assert out == [
        ("positive", "b"),
        ("matched_results", ["b"]),
        ("matched_count", 1),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["(", "[a-"])
async def test_extract_rejects_invalid_pattern(pattern):
    block = ExtractTextInformationBlock()
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await _outputs(block, block.Input(text="abc", pattern=pattern))


@pytest.mark.asyncio
async def test_extract_rejects_invalid_dangerous_pattern():
    # Dangerous-looking patterns go through the `regex` module; a malformed
    # one must still surface instead of reading as "no match".
    block = ExtractTextInformationBlock()
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        await _outputs(block, block.Input(text="aaa", pattern="(a+)+("))


@pytest.mark.asyncio
async def test_extract_timeout_still_returns_empty_results(mocker):
    def _timeout(*args, **kwargs):
        raise TimeoutError("regex timed out")

    mocker.patch("backend.blocks.text.regex.finditer", side_effect=_timeout)
    block = ExtractTextInformationBlock()
    out = await _outputs(block, block.Input(text="aaaa", pattern="(a+)+"))
    assert out == [
        ("negative", "aaaa"),
        ("matched_results", []),
        ("matched_count", 0),
    ]


# --- #15258 ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [0, False, "", [], {}])
async def test_match_forwards_falsy_data(data):
    block = MatchTextPatternBlock()
    positive = await _outputs(block, block.Input(text="hello", match="hell", data=data))
    negative = await _outputs(block, block.Input(text="hello", match="xyz", data=data))
    assert positive == [("positive", data)]
    assert negative == [("negative", data)]


@pytest.mark.asyncio
async def test_match_falls_back_to_text_when_data_unconnected():
    block = MatchTextPatternBlock()
    out = await _outputs(block, block.Input(text="hello", match="hell"))
    assert out == [("positive", "hello")]
