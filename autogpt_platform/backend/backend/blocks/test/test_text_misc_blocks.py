"""Regression tests for #15262 (CreateListBlock empty leading chunk),
#15273 (TextDecoderBlock mojibake) and #15274 (CodeExtractionBlock fence
removal is case-sensitive)."""

import pytest

from backend.blocks.code_extraction_block import CodeExtractionBlock
from backend.blocks.data_manipulation import CreateListBlock
from backend.blocks.decoder_block import TextDecoderBlock


async def _outputs(block, input_data) -> list[tuple[str, object]]:
    return [(name, value) async for name, value in block.run(input_data)]


# --- #15262 ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_list_no_empty_chunk_when_first_value_exceeds_max_tokens():
    block = CreateListBlock()
    big = "x" * 5000
    out = await _outputs(block, block.Input(values=[big, "b"], max_tokens=10))
    assert out == [("list", [big]), ("list", ["b"])]


@pytest.mark.asyncio
async def test_create_list_empty_input_still_yields_one_empty_list():
    block = CreateListBlock()
    out = await _outputs(block, block.Input(values=[]))
    assert out == [("list", [])]


# --- #15273 ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("café\\nbar", "café\nbar"),
        ("日本語\\n", "日本語\n"),
        ("👋\\tüber", "👋\tüber"),
        (
            'Hello\\nWorld!\\nThis is a \\"quoted\\" string.',
            'Hello\nWorld!\nThis is a "quoted" string.',
        ),
    ],
)
async def test_text_decoder_keeps_non_ascii_text(text, expected):
    block = TextDecoderBlock()
    out = await _outputs(block, block.Input(text=text))
    assert out == [("decoded_text", expected)]


@pytest.mark.asyncio
async def test_text_decoder_invalid_escape_is_a_value_error():
    block = TextDecoderBlock()
    with pytest.raises(ValueError, match="invalid escape sequence"):
        await _outputs(block, block.Input(text="C:\\Users\\xavier"))


# --- #15274 ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["python", "Python", "PYTHON"])
async def test_code_extraction_removes_fence_regardless_of_case(fence):
    block = CodeExtractionBlock()
    text = f"Intro\n```{fence}\nprint(1)\n```\nOutro"
    out = dict(await _outputs(block, block.Input(text=text)))
    assert out["python"] == "print(1)"
    assert out["remaining_text"] == "Intro\nOutro"
