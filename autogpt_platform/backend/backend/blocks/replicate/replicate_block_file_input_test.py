"""Unit tests for ReplicateModelBlock file-input handling.

Verifies that uploaded ``files`` are merged into ``model_inputs`` under
``file_input_field`` so AutoPilot can pass file references instead of inlining
base64 into the inputs dict by hand:

1. No files → model_inputs passed through unchanged (and not mutated in place)
2. Single file → bound as a single data URI under the default field ``image``
3. Multiple files → bound as a list of data URIs
4. Custom ``file_input_field`` → files bound under the chosen name
5. files take precedence over a same-named key in model_inputs
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.replicate._auth import TEST_CREDENTIALS_INPUT
from backend.blocks.replicate.replicate_block import ReplicateModelBlock
from backend.data.execution import ExecutionContext


def _make_input(**overrides):
    base = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "model_name": "owner/model",
    }
    base.update(overrides)
    return ReplicateModelBlock.Input(**base)


def _exec_context():
    return ExecutionContext(user_id="user-1", graph_exec_id="exec-1")


@pytest.mark.asyncio
async def test_no_files_returns_inputs_unchanged():
    block = ReplicateModelBlock()
    original = {"prompt": "hi", "num_outputs": 1}
    input_data = _make_input(model_inputs=dict(original))

    result = await block._build_model_inputs(input_data, _exec_context())

    assert result == original
    # The returned dict must be a copy, not the schema's own dict.
    assert result is not input_data.model_inputs


@pytest.mark.asyncio
async def test_single_file_bound_as_string_under_default_field():
    block = ReplicateModelBlock()
    input_data = _make_input(
        model_inputs={"prompt": "describe"},
        files=["workspace://abc"],
    )

    with patch(
        "backend.blocks.replicate.replicate_block.store_media_file",
        new=AsyncMock(return_value="data:image/png;base64,AAAA"),
    ):
        result = await block._build_model_inputs(input_data, _exec_context())

    assert result == {
        "prompt": "describe",
        "image": "data:image/png;base64,AAAA",
    }


@pytest.mark.asyncio
async def test_multiple_files_bound_as_list():
    block = ReplicateModelBlock()
    input_data = _make_input(
        files=["workspace://a", "workspace://b"],
    )

    with patch(
        "backend.blocks.replicate.replicate_block.store_media_file",
        new=AsyncMock(
            side_effect=["data:image/png;base64,A", "data:image/png;base64,B"]
        ),
    ):
        result = await block._build_model_inputs(input_data, _exec_context())

    assert result["image"] == [
        "data:image/png;base64,A",
        "data:image/png;base64,B",
    ]


@pytest.mark.asyncio
async def test_custom_file_input_field():
    block = ReplicateModelBlock()
    input_data = _make_input(
        files=["workspace://a"],
        file_input_field="input_image",
    )

    with patch(
        "backend.blocks.replicate.replicate_block.store_media_file",
        new=AsyncMock(return_value="data:image/png;base64,Z"),
    ):
        result = await block._build_model_inputs(input_data, _exec_context())

    assert result == {"input_image": "data:image/png;base64,Z"}


@pytest.mark.asyncio
async def test_files_override_same_named_model_input():
    block = ReplicateModelBlock()
    input_data = _make_input(
        model_inputs={"image": "stale-value"},
        files=["workspace://a"],
    )

    with patch(
        "backend.blocks.replicate.replicate_block.store_media_file",
        new=AsyncMock(return_value="data:image/png;base64,NEW"),
    ):
        result = await block._build_model_inputs(input_data, _exec_context())

    assert result["image"] == "data:image/png;base64,NEW"
