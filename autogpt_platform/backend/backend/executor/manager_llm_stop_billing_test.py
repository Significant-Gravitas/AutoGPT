"""A stop that cuts off an AI block's model request bills that request.

The provider has already been sent the prompt, so a stop request mid-call
settles the node on the request's estimated input tokens, at the model's input
rate. The block runs for real through the executor and its billing, against
the real block costs; only the provider, the database and the credential store
are replaced.
"""

import asyncio
import math
import threading
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data.block_cost_config import TOKEN_COST
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.executor import billing
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.credentials_store import (
    anthropic_credentials,
    open_router_credentials,
)

NODE_EXEC_ID = "node-exec-1"
PRE_FLIGHT = 5
# A token-priced model, and an OpenRouter one priced by provider cost.
PRICED_MODELS = [
    pytest.param(LLMModel.CLAUDE_4_5_HAIKU, anthropic_credentials, id="tokens"),
    pytest.param(LLMModel("x-ai/grok-4"), open_router_credentials, id="cost-usd"),
]


class _Run:
    """The node's collaborators, with a provider that takes the request and
    never replies."""

    def __init__(self) -> None:
        self.request_sent = asyncio.Event()
        self.reconcile = AsyncMock(side_effect=billing.charge_reconciled_usage)
        self.spent: list[int] = []
        self.statuses: list[ExecutionStatus] = []
        self.node = Node(id="node-1", block_id=AITextGeneratorBlock().id)

    async def provider(self, **_: Any) -> None:
        self.request_sent.set()
        await asyncio.Event().wait()

    async def spend(self, *, cost: int, **_: Any) -> int:
        self.spent.append(cost)
        return 0

    async def update_status(self, *, status: ExecutionStatus, **_: Any) -> None:
        self.statuses.append(status)

    def patches(self, credentials: APIKeyCredentials):
        mock_key = credentials.model_copy(update={"api_key": SecretStr("mock-key")})
        db_client = MagicMock(
            get_node=AsyncMock(return_value=self.node),
            upsert_execution_output=AsyncMock(),
            get_node_execution=AsyncMock(return_value=None),
        )
        wallet = MagicMock(spend_credits=AsyncMock(side_effect=self.spend))
        return [
            patch("backend.util.llm.providers.call_provider", self.provider),
            patch(
                "backend.executor.billing.get_database_manager_async_client",
                return_value=wallet,
            ),
            patch(
                "backend.executor.manager.get_db_async_client", return_value=db_client
            ),
            patch(
                "backend.executor.manager._get_node_credentials",
                AsyncMock(return_value=(mock_key, None)),
            ),
            patch(
                "backend.executor.manager.acquire_auto_credentials",
                AsyncMock(return_value=({}, [])),
            ),
            patch("backend.executor.manager.log_system_credential_cost", AsyncMock()),
            patch(
                "backend.executor.manager.billing.charge_reconciled_usage",
                self.reconcile,
            ),
            patch(
                "backend.executor.manager.async_update_node_execution_status",
                self.update_status,
            ),
            patch(
                "backend.executor.manager.async_update_graph_execution_state",
                AsyncMock(),
            ),
        ]


def _node_exec(
    block_id: str, model: LLMModel, credentials: APIKeyCredentials
) -> NodeExecutionEntry:
    graph_exec_id = str(uuid.uuid4())
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=graph_exec_id,
        graph_id="graph-1",
        graph_version=1,
        node_exec_id=NODE_EXEC_ID,
        node_id="node-1",
        block_id=block_id,
        inputs={
            "prompt": "Write a long story about a lighthouse. " * 2000,
            "model": model.value,
            "credentials": {
                "id": credentials.id,
                "provider": credentials.provider,
                "type": credentials.type,
                "title": credentials.title,
            },
        },
        execution_context=ExecutionContext(root_execution_id=graph_exec_id),
        pre_flight_charge=PRE_FLIGHT,
    )


async def _stop_mid_call(
    model: LLMModel,
    credentials: APIKeyCredentials,
    *,
    stop_requested: bool,
    extra_patches: tuple = (),
) -> _Run:
    run = _Run()
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()
    progress = NodeExecutionProgress()

    patches = [*run.patches(credentials), *extra_patches]
    for p in patches:
        p.start()
    try:
        node_run = asyncio.ensure_future(
            processor.on_node_execution(
                node_exec=_node_exec(run.node.block_id, model, credentials),
                node_exec_progress=progress,
                nodes_input_masks=None,
                graph_stats_pair=(GraphExecutionStats(), threading.Lock()),
            )
        )
        progress.add_task(NODE_EXEC_ID, node_run)
        await asyncio.wait_for(run.request_sent.wait(), timeout=60)
        progress.stop(stop_requested=stop_requested)
        try:
            await asyncio.wait_for(node_run, timeout=60)
        except asyncio.CancelledError:
            pass
    finally:
        for p in patches:
            p.stop()
    return run


def _estimated_input(run: _Run) -> int:
    [settlement] = run.reconcile.await_args_list
    return settlement.kwargs["stats"].abandoned_input_token_count


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "credentials"), PRICED_MODELS)
async def test_a_stop_mid_call_bills_the_requests_estimated_input(model, credentials):
    run = await _stop_mid_call(model, credentials, stop_requested=True)

    assert run.statuses[-1] == ExecutionStatus.TERMINATED
    estimated_input = _estimated_input(run)
    assert estimated_input > 10_000
    estimate_cost = math.ceil(estimated_input * TOKEN_COST[model].input / 1e6)
    assert run.spent == [estimate_cost - PRE_FLIGHT]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "credentials"), PRICED_MODELS)
async def test_a_cancel_without_a_stop_request_settles_on_reported_usage(
    model, credentials
):
    """Cancelled by the run ending another way, the node settles on the usage
    the provider reported, which is none here."""
    run = await _stop_mid_call(model, credentials, stop_requested=False)

    assert run.statuses[-1] == ExecutionStatus.TERMINATED
    assert run.spent == [-PRE_FLIGHT]


@pytest.mark.asyncio
async def test_a_second_cancel_while_the_node_unwinds_keeps_its_usage():
    """The cancel lands again while the node releases its credentials; the
    node is still billed the usage its block recorded."""
    model, credentials = LLMModel.CLAUDE_4_5_HAIKU, anthropic_credentials
    second_cancel = patch(
        "backend.executor.manager._release_credential_leases",
        AsyncMock(side_effect=asyncio.CancelledError),
    )

    run = await _stop_mid_call(
        model, credentials, stop_requested=True, extra_patches=(second_cancel,)
    )

    estimated_input = _estimated_input(run)
    assert estimated_input > 10_000
    estimate_cost = math.ceil(estimated_input * TOKEN_COST[model].input / 1e6)
    assert run.spent == [estimate_cost - PRE_FLIGHT]
