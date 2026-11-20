from unittest.mock import patch

import pytest
from pydantic import SecretStr

from backend.blocks.stagehand.blocks import (
    StagehandActBlock,
    StagehandExtractBlock,
    StagehandObserveBlock,
    StagehandRecommendedLLMModel,
)
from backend.data.execution import ExecutionContext
from backend.data.model import APIKeyCredentials

# Anthropic retires this model on 2026-11-30; it left Stagehand's list.
SONNET_4_5 = "claude-sonnet-4-5-20250929"

STAGEHAND_CREDENTIALS = APIKeyCredentials(
    provider="stagehand", api_key=SecretStr("stagehand-key"), title="Stagehand"
)
ANTHROPIC_CREDENTIALS = APIKeyCredentials(
    provider="anthropic", api_key=SecretStr("anthropic-key"), title="Anthropic"
)


def saved_inputs(block: type, model: str) -> dict:
    """A Stagehand node's stored inputs, as the executor passes them in."""
    data = {
        "stagehand_credentials": {
            "provider": "stagehand",
            "id": STAGEHAND_CREDENTIALS.id,
            "type": "api_key",
            "title": "Stagehand",
        },
        "browserbase_project_id": "project",
        "model": model,
        "model_credentials": {
            "provider": "anthropic",
            "id": ANTHROPIC_CREDENTIALS.id,
            "type": "api_key",
            "title": "Anthropic",
        },
        "url": "https://example.com",
    }
    if block is StagehandActBlock:
        return {**data, "action": ["click the first link"]}
    return {**data, "instruction": "find the first link"}


@pytest.mark.parametrize(
    "block", [StagehandObserveBlock, StagehandActBlock, StagehandExtractBlock]
)
def test_a_node_saved_on_claude_sonnet_4_5_passes_input_validation(block: type):
    """The executor checks a node's inputs against the JSON schema before the
    ``model`` field resolves them, and the schema no longer lists Sonnet 4.5.
    The check reads the saved value as Sonnet 4.6, on a copy."""
    data = saved_inputs(block, SONNET_4_5)

    assert block.Input.validate_data(data) is None
    assert data["model"] == SONNET_4_5


def test_a_model_stagehand_never_offered_is_still_refused():
    data = saved_inputs(StagehandObserveBlock, "gpt-5-2025-08-07")

    assert StagehandObserveBlock.Input.validate_data(data) is not None


@pytest.mark.asyncio
async def test_a_node_saved_on_claude_sonnet_4_5_runs_on_sonnet_4_6():
    """Through the real execute path: input validation, then the input model,
    which resolves the saved value to Sonnet 4.6."""
    block = StagehandObserveBlock()
    models: list[StagehandRecommendedLLMModel] = []

    async def run(input_data, **kwargs):
        models.append(input_data.model)
        yield "selector", "//a[1]"

    with patch.object(block, "run", run):
        outputs = [
            output
            async for output in block.execute(
                saved_inputs(StagehandObserveBlock, SONNET_4_5),
                execution_context=ExecutionContext(user_id="user"),
                stagehand_credentials=STAGEHAND_CREDENTIALS,
                model_credentials=ANTHROPIC_CREDENTIALS,
            )
        ]

    assert outputs == [("selector", "//a[1]")]
    assert models == [StagehandRecommendedLLMModel.CLAUDE_4_6_SONNET]
    assert models[0].provider_name == "anthropic/claude-sonnet-4-6"


def test_the_model_list_no_longer_offers_claude_sonnet_4_5():
    assert SONNET_4_5 not in {model.value for model in StagehandRecommendedLLMModel}
