import pytest

from backend.blocks.encoder_block import TextEncoderBlock


@pytest.mark.asyncio
async def test_text_encoder_basic():
    """Test basic encoding of newlines and special characters."""
    block = TextEncoderBlock()
    result = []
    async for output in block.run(TextEncoderBlock.Input(text="Hello\nWorld")):
        result.append(output)

    assert len(result) == 1
    assert result[0][0] == "encoded_text"
    assert result[0][1] == "Hello\\nWorld"


@pytest.mark.asyncio
async def test_text_encoder_multiple_escapes():
    """Test encoding of multiple escape sequences."""
    block = TextEncoderBlock()
    result = []
    async for output in block.run(
        TextEncoderBlock.Input(text="Line1\nLine2\tTabbed\rCarriage")
    ):
        result.append(output)

    assert len(result) == 1
    assert result[0][0] == "encoded_text"
    assert "\\n" in result[0][1]
    assert "\\t" in result[0][1]
    assert "\\r" in result[0][1]


@pytest.mark.asyncio
async def test_text_encoder_unicode():
    """Test that unicode characters are handled correctly."""
    block = TextEncoderBlock()
    result = []
    async for output in block.run(TextEncoderBlock.Input(text="Hello 世界\n")):
        result.append(output)

    assert len(result) == 1
    assert result[0][0] == "encoded_text"
    # Unicode characters should be escaped as \uXXXX sequences
    assert "\\n" in result[0][1]


@pytest.mark.asyncio
async def test_text_encoder_empty_string():
    """Test encoding of an empty string."""
    block = TextEncoderBlock()
    result = []
    async for output in block.run(TextEncoderBlock.Input(text="")):
        result.append(output)

    assert len(result) == 1
    assert result[0][0] == "encoded_text"
    assert result[0][1] == ""


@pytest.mark.asyncio
async def test_text_encoder_error_handling():
    """Test that encoding errors are handled gracefully."""
    from unittest.mock import patch

    block = TextEncoderBlock()
    result = []

    with patch("codecs.encode", side_effect=Exception("Mocked encoding error")):
        async for output in block.run(TextEncoderBlock.Input(text="test")):
            result.append(output)

    assert len(result) == 1
    assert result[0][0] == "error"
    assert "Mocked encoding error" in result[0][1]
