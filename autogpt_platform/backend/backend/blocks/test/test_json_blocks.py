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
async def test_json_encoder_error():
    block = JSONEncoderBlock()

    data = {}
    data["self"] = data

    with pytest.raises(ValueError, match="JSON Encoding Error") as exc_info:
        async for output in block.run(JSONEncoderBlock.Input(data=data)):
            pass

    error_message = str(exc_info.value)
    assert "JSON Encoding Error" in error_message


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

    with pytest.raises(ValueError, match="JSON Decoding Error") as exc_info:
        async for _ in block.run(JSONDecoderBlock.Input(json_str=json_str)):
            pass
    error_message = str(exc_info.value)
    assert "JSON Decoding Error" in error_message


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
async def test_json_encoder_invalid_floats(invalid_float):
    block = JSONEncoderBlock()
    data = {"invalid": invalid_float}

    try:
        outputs = []

        async for output in block.run(JSONEncoderBlock.Input(data=data)):
            outputs.append(output)

        assert len(outputs) == 1
        name, value = outputs[0]
        assert name == "json_str"
        # Ensure produced JSON is valid
        json.loads(value)
    except ValueError as e:
        # Also acceptable if serializer rejects invalid floats
        assert "JSON Encoding Error" in str(e)
