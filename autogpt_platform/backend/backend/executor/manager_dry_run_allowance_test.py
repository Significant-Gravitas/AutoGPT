"""A dry run's block that runs for real on the platform's key is checked and metered."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.llm_registry import llm_models
from backend.data.model import NodeExecutionStats
from backend.executor.manager import execute_node
from backend.integrations.credentials_store import openai_credentials
from backend.util.llm.providers import ProviderResponse

_MOD = "backend.executor.manager"


@pytest.mark.asyncio
async def test_a_spent_allowance_stops_the_block_before_it_runs(dry_run_boundaries):
    refusal, track = dry_run_boundaries
    refusal.return_value = "You've reached your daily usage limit."
    block = _block()

    outputs = [item async for item in execute_node(*_args(block))]

    assert outputs == [("error", "You've reached your daily usage limit.")]
    block.ran.assert_not_called()
    refusal.assert_awaited_once_with("user-1", surface="block")
    track.assert_not_awaited()


@pytest.mark.asyncio
async def test_what_the_block_spent_is_recorded_against_the_user(dry_run_boundaries):
    _, track = dry_run_boundaries
    block = _block()

    outputs = [item async for item in execute_node(*_args(block))]

    assert outputs == [("response", "ok")]
    track.assert_awaited_once()
    assert track.await_args.args == (block.execution_stats,)
    assert track.await_args.kwargs["user_id"] == "user-1"
    assert track.await_args.kwargs["model"] == "sim-model"


@pytest.mark.asyncio
async def test_the_record_names_the_model_the_block_started_on(
    dry_run_boundaries, monkeypatch: pytest.MonkeyPatch
):
    """A block the dry run runs for real starts at 23:59:50 on the eve of its
    model's shutdown date, and midnight passes while the provider has its
    request. The record names the model it ran, not the fallback a block
    starting after midnight runs."""
    _, track = dry_run_boundaries
    clock = {"today": date(2026, 12, 10)}
    monkeypatch.setattr(llm_models, "_today", lambda: clock["today"])
    asked: list[str] = []

    async def provider(*, model: str, **_: object) -> ProviderResponse:
        asked.append(model)
        clock["today"] = date(2026, 12, 11)
        return ProviderResponse(content="Hello.", prompt_tokens=10, completion_tokens=1)

    platform_key = openai_credentials.model_copy(
        update={"api_key": SecretStr("mock-key")}
    )
    with (
        patch(
            f"{_MOD}.prepare_dry_run",
            side_effect=lambda _block, data: {
                **data,
                "prompt": "Say hello.",
                "model": LLMModel.GPT5_MINI.value,
            },
        ),
        patch(f"{_MOD}.get_dry_run_credentials", return_value=platform_key),
        patch("backend.util.llm.providers.call_provider", provider),
    ):
        outputs = [item async for item in execute_node(*_args(AITextGeneratorBlock()))]

    assert ("response", "Hello.") in outputs
    assert asked == [LLMModel.GPT5_MINI.value]
    track.assert_awaited_once()
    assert track.await_args.kwargs["model"] == LLMModel.GPT5_MINI.value


@pytest.fixture
def dry_run_boundaries():
    scope = MagicMock(_user=None, _tags={})
    refusal = AsyncMock(return_value=None)
    track = AsyncMock()
    with (
        patch(
            f"{_MOD}.validate_exec",
            side_effect=lambda _node, data, **_kwargs: (data, None),
        ),
        patch(f"{_MOD}._sentry_get_current_scope", return_value=scope),
        patch(
            f"{_MOD}.prepare_dry_run",
            side_effect=lambda _block, data: {**data, "model": "sim-model"},
        ),
        patch(f"{_MOD}.get_dry_run_credentials", return_value=MagicMock()),
        patch(f"{_MOD}.platform_spend_refusal", refusal),
        patch(f"{_MOD}.track_dry_run_block_cost", track),
    ):
        yield refusal, track


def _block() -> MagicMock:
    input_schema = MagicMock()
    input_schema.get_credentials_fields.return_value = {}
    input_schema.get_credentials_fields_info.return_value = {}
    input_schema.get_auto_credentials_fields.return_value = {}
    block = MagicMock()
    block.id = "block-1"
    block.name = "PlatformKeyBlock"
    block.disabled = False
    block.input_schema = input_schema
    block.execution_stats = NodeExecutionStats(input_token_count=10)
    block.with_retired_models_replaced.side_effect = lambda input_data, _model: (
        input_data
    )
    block.ran = MagicMock()

    async def execute(_input_data, **_kwargs):
        block.ran()
        yield "response", "ok"

    block.execute = execute
    return block


def _args(block: object):
    node = MagicMock()
    node.block = block
    node.input_default = {}
    entry = NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="graph-exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(dry_run=True),
    )
    return node, entry, SimpleNamespace(creds_manager=MagicMock())
