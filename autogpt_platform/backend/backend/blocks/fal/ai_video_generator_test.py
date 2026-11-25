from unittest.mock import patch

import pytest

from backend.blocks.fal._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock, FalModel
from backend.data.execution import ExecutionContext
from backend.data.graph import Node
from backend.executor.utils import validate_exec

# FAL removed both; its queue answers 404 for each.
MOCHI = "fal-ai/mochi-v1"
LUMA_DREAM_MACHINE = "fal-ai/luma-dream-machine"

EXECUTION_IDS = {
    "user_id": "user",
    "graph_id": "graph",
    "graph_version": 1,
    "graph_exec_id": "graph-exec",
    "node_id": "node",
    "node_exec_id": "node-exec",
}


def test_the_block_offers_only_models_fal_serves():
    schema = AIVideoGeneratorBlock.Input.jsonschema()["properties"]["model"]

    assert {MOCHI, LUMA_DREAM_MACHINE}.isdisjoint(schema["enum"])
    assert schema["default"] == FalModel.LTX_2_3_FAST.value
    assert AIVideoGeneratorBlock.Input.validate_data(saved_inputs("fal-ai/not-a-model"))


@pytest.mark.parametrize("saved_model", [MOCHI, LUMA_DREAM_MACHINE])
@pytest.mark.asyncio
async def test_a_node_saved_on_a_retired_model_runs_on_ltx_2_3_fast(saved_model: str):
    """The executor's input check accepts the saved value and queues the
    replacement, and the block runs on it; the stored data keeps its value."""
    data = saved_inputs(saved_model)
    block = AIVideoGeneratorBlock()
    models: list[FalModel] = []

    async def generate_video(input_data, credentials) -> str:
        models.append(input_data.model)
        return "data:video/mp4;base64,AAAA"

    queued, error = validate_exec(
        Node(block_id=block.id, input_default=data), {}, resolve_input=False
    )
    with patch.object(block, "generate_video", generate_video):
        outputs = [
            output
            async for output in block.execute(
                data,
                **EXECUTION_IDS,
                execution_context=ExecutionContext(**EXECUTION_IDS),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert queued is not None, error
    assert queued["model"] == FalModel.LTX_2_3_FAST
    assert models == [FalModel.LTX_2_3_FAST]
    assert [name for name, _ in outputs] == ["video_url"]
    assert data["model"] == saved_model


def saved_inputs(model: str) -> dict:
    """A video node's stored inputs, as the executor passes them in."""
    return {
        "prompt": "A paper boat drifting down a rainy street.",
        "model": model,
        "credentials": TEST_CREDENTIALS_INPUT,
    }
