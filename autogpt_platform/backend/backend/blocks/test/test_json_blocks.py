import json

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
    parsed = json.loads(value)
    assert parsed["hello"] == "world"
    assert parsed["numbers"] == [1, 2, 3]
    assert parsed["nested"]["key"] is True


@pytest.mark.asyncio
async def test_json_encoder_circular_reference():
    block = JSONEncoderBlock()

    data = {}
    data["self"] = data

    with pytest.raises(ValueError, match="JSON Encoding Error"):
        async for _ in block.run(JSONEncoderBlock.Input(data=data)):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_data",
    [
        "a string",
        42,
        None,
        [1, 2, "three"],
        {},
        [],
        "",
    ],
)
async def test_json_encoder_primitives(input_data):
    block = JSONEncoderBlock()

    outputs = []
    async for output in block.run(JSONEncoderBlock.Input(data=input_data)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "json_str"
    # Compare semantic equality instead of raw formatting
    assert json.loads(value) == input_data


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

    with pytest.raises(ValueError, match="JSON Decoding Error"):
        async for _ in block.run(JSONDecoderBlock.Input(json_str=json_str)):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "json_str,expected_data",
    [
        ('"a string"', "a string"),
        ("42", 42),
        ("null", None),
        ('[1, 2, "three"]', [1, 2, "three"]),
        ("{}", {}),
        ("[]", []),
        ('""', ""),
    ],
)
async def test_json_decoder_primitives(json_str, expected_data):
    block = JSONDecoderBlock()

    outputs = []

    async for output in block.run(JSONDecoderBlock.Input(json_str=json_str)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "data"
    assert value == expected_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_float",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
async def test_json_encoder_non_finite_floats_become_null(invalid_float):
    block = JSONEncoderBlock()
    data = {"invalid": invalid_float}

    outputs = []
    async for output in block.run(JSONEncoderBlock.Input(data=data)):
        outputs.append(output)

    assert len(outputs) == 1
    name, value = outputs[0]
    assert name == "json_str"
    parsed = json.loads(value)
    assert parsed["invalid"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_data",
    [
        {"name": "AutoGPT", "active": True},
        [1, 2, 3, "test"],
        "simple string",
        42,
        None,
        {"nested": {"list": [1, 2, None]}},
        [],
        {},
    ],
)
async def test_json_roundtrip(original_data):
    encoder = JSONEncoderBlock()
    decoder = JSONDecoderBlock()

    encoder_outputs = []
    async for output in encoder.run(JSONEncoderBlock.Input(data=original_data)):
        encoder_outputs.append(output)

    assert len(encoder_outputs) == 1
    name, json_str = encoder_outputs[0]
    assert name == "json_str"

    decoder_outputs = []
    async for output in decoder.run(JSONDecoderBlock.Input(json_str=json_str)):
        decoder_outputs.append(output)

    assert len(decoder_outputs) == 1
    name, decoded_data = decoder_outputs[0]
    assert name == "data"
    assert decoded_data == original_data
