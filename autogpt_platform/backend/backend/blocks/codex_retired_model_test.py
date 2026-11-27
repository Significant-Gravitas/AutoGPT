from unittest.mock import patch

import pytest

from backend.blocks.codex import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    CodeGenerationBlock,
    CodexCallResult,
    CodexModel,
)
from backend.copilot.tools.helpers import _pre_run_charge
from backend.data.execution import ExecutionContext
from backend.data.graph import Node
from backend.executor.utils import block_usage_cost, validate_exec
from backend.integrations.credentials_store import openai_credentials

# OpenAI shut this model down on 2026-07-23; it left the Codex block's list.
GPT_5_1_CODEX = "gpt-5.1-codex"

PLATFORM_OPENAI_KEY = {
    "provider": openai_credentials.provider,
    "id": openai_credentials.id,
    "type": openai_credentials.type,
    "title": openai_credentials.title,
}


def saved_inputs(model: str, credentials: dict = TEST_CREDENTIALS_INPUT) -> dict:
    """A Codex node's stored inputs, as the executor passes them in."""
    return {
        "prompt": "Write a function that reverses a list.",
        "model": model,
        "credentials": credentials,
    }


def test_the_codex_block_no_longer_offers_gpt_5_1_codex():
    schema = CodeGenerationBlock.Input.jsonschema()["properties"]["model"]

    assert GPT_5_1_CODEX not in {model.value for model in CodexModel}
    assert GPT_5_1_CODEX not in schema["enum"]


@pytest.mark.asyncio
async def test_a_node_saved_on_gpt_5_1_codex_validates_and_runs_on_gpt_5_3_codex():
    """The block's JSON schema no longer lists the saved value, so the input
    check reads it as gpt-5.3-codex, on a copy, and the ``model`` field
    resolves it the same way. A model the block never offered is still
    refused."""
    data = saved_inputs(GPT_5_1_CODEX)
    block = CodeGenerationBlock()
    models: list[CodexModel] = []

    async def call_codex(*, model: CodexModel, **_kwargs) -> CodexCallResult:
        models.append(model)
        return CodexCallResult(response="done", reasoning="", response_id="r-1")

    with patch.object(block, "call_codex", call_codex):
        outputs = [
            output
            async for output in block.execute(
                data,
                execution_context=ExecutionContext(user_id="user"),
                credentials=TEST_CREDENTIALS,
            )
        ]

    assert CodeGenerationBlock.Input.validate_data(data) is None
    assert data["model"] == GPT_5_1_CODEX
    assert models == [CodexModel.GPT5_3_CODEX]
    assert ("response", "done") in outputs
    assert CodeGenerationBlock.Input.validate_data(saved_inputs("gpt-6.1-sol"))


def test_a_node_saved_on_gpt_5_1_codex_is_billed_as_gpt_5_3_codex():
    """The executor queues the node's input coerced through its schema, so
    the charge matches gpt-5.3-codex's cost entry, not none at all."""
    block = CodeGenerationBlock()
    node = Node(
        block_id=block.id,
        input_default=saved_inputs(GPT_5_1_CODEX, credentials=PLATFORM_OPENAI_KEY),
    )

    queued, error = validate_exec(node, {}, resolve_input=False)

    assert queued is not None, error
    assert queued["model"] == CodexModel.GPT5_3_CODEX
    _, cost_filter = block_usage_cost(block, queued)
    assert cost_filter["model"] == CodexModel.GPT5_3_CODEX


def test_a_raw_saved_gpt_5_1_codex_is_priced_as_gpt_5_3_codex():
    """The gate's run estimate (``graph_cost_credits``) and the expert spend
    check (``check_spend_approval``) price a node's input before its schema
    reads the value, so the retired value needs a cost entry of its own, or
    the run is priced at nothing and skips the spend check."""
    block = CodeGenerationBlock()
    raw = saved_inputs(GPT_5_1_CODEX, credentials=PLATFORM_OPENAI_KEY)
    current = saved_inputs("gpt-5.3-codex", credentials=PLATFORM_OPENAI_KEY)

    cost, cost_filter = block_usage_cost(block, raw)

    # Pricing coerces the input as the run does, so this is the model that runs.
    assert cost_filter["model"] == CodexModel.GPT5_3_CODEX
    assert cost == block_usage_cost(block, current)[0]
    assert _pre_run_charge(block, raw)[0] == _pre_run_charge(block, current)[0] > 0
