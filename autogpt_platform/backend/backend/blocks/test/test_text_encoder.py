import pytest

from backend.blocks.decoder_block import TextEncoderBlock


@pytest.mark.asyncio
async def test_text_encoder_error():
    block = TextEncoderBlock()
    # It's hard to force a unicode_escape error with a simple string as Python handles almost everything.
    # We can try to mock codecs.encode to raise an exception to verify the error handling path.
    from unittest.mock import patch

    with patch("codecs.encode", side_effect=Exception("Mocked encoding error")):
        gen = block.run(TextEncoderBlock.Input(text="test"))
        output = []
        async for o in gen:
            output.append(o)

        assert len(output) == 1
        assert output[0][0] == "error"
        assert "Mocked encoding error" in output[0][1]


@pytest.mark.asyncio
async def test_text_encoder_success():
    block = TextEncoderBlock()
    gen = block.run(TextEncoderBlock.Input(text="Hello\nWorld"))
    output = []
    async for o in gen:
        output.append(o)

    assert len(output) == 1
    assert output[0][0] == "encoded_text"
    assert output[0][1] == "Hello\\nWorld"
