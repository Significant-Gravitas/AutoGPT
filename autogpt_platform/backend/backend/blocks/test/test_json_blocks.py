import pytest

from backend.blocks.json_blocks import JSONDecoderBlock, JSONEncoderBlock


@pytest.mark.asyncio
async def test_json_encoder_success():
    block = JSONEncoderBlock()
    data = {"hello": "world", "numbers": [1, 2, 3], "nested": {"key": True}}

    outputs = []
    async for output in block.run(JSONEncoderBlock.Input(data=data)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "json_str"
    assert "hello" in value
    assert "world" in value
    assert "numbers" in value


@pytest.mark.asyncio
async def test_json_encoder_error():
    block = JSONEncoderBlock()
    data = {"set_value": {1, 2, 3}}

    outputs = []
    async for output in block.run(JSONEncoderBlock.Input(data=data)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "error"
    assert "JSON Encoding Error" in value


@pytest.mark.asyncio
async def test_json_decoder_success():
    block = JSONDecoderBlock()
    json_str = '{"a": 1, "b": [true, false, null], "c": {"d": "nested"}}'

    outputs = []
    async for output in block.run(JSONDecoderBlock.Input(json_str=json_str)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "data"
    assert value == {"a": 1, "b": [True, False, None], "c": {"d": "nested"}}


@pytest.mark.asyncio
async def test_json_decoder_error():
    block = JSONDecoderBlock()
    # Malformed JSON string missing closing bracket
    json_str = '{"a": 1, "b": [true, false,'

    outputs = []
    async for output in block.run(JSONDecoderBlock.Input(json_str=json_str)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "error"
    assert "JSON Decoding Error" in value
