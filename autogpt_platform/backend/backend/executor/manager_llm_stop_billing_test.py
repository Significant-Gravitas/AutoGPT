"""A stop that cuts off an AI block's model request bills that request.

The provider has already been sent the prompt, so a stop request mid-call
settles the node on the request's estimated input tokens, at the model's input
rate. So does any other ending its user brings about mid-call, such as a node
the balance can't pay for. The block runs for real through the executor and its
billing, against the real block costs; only the provider, the database and the
credential store are replaced.
"""

import asyncio
import math
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data.block_cost_config import TOKEN_COST
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.graph import Node
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.executor import billing
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.credentials_store import (
    anthropic_credentials,
    open_router_credentials,
)
from backend.util.exceptions import InsufficientBalanceError, ModerationError

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


def _fail_mid_call(
    model: LLMModel,
    credentials: APIKeyCredentials,
    error: Exception,
    output_refusal: ModerationError | None = None,
) -> _Run:
    """A run of two nodes through the executor's own dispatch loop and cleanup:
    the first is charged and mid-call when charging the second raises
    ``error``, which fails the run; output moderation then returns
    ``output_refusal``."""
    run = _Run()
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()
    mid_call = _node_exec(run.node.block_id, model, credentials)
    uncharged = mid_call.model_copy(
        update={"node_exec_id": "node-exec-2", "node_id": "node-2"}
    )

    def charge(node_exec: NodeExecutionEntry, **_: Any) -> tuple[int, int, int]:
        if node_exec is mid_call:
            return PRE_FLIGHT, 0, PRE_FLIGHT
        deadline = time.monotonic() + 60
        while not run.request_sent.is_set():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raise error

    db_client = MagicMock()
    db_client.get_credits.return_value = PRE_FLIGHT
    db_client.get_node.return_value = run.node
    db_client.get_node_executions.return_value = [
        SimpleNamespace(
            status=ExecutionStatus.QUEUED,
            queue_time=None,
            to_node_execution_entry=lambda _context, entry=entry: entry,
        )
        for entry in (mid_call, uncharged)
    ]
    automod = MagicMock(
        moderate_graph_execution_inputs=AsyncMock(return_value=None),
        moderate_graph_execution_outputs=AsyncMock(return_value=output_refusal),
        update_failed_nodes_for_moderation=AsyncMock(),
    )
    patches = [
        *run.patches(credentials),
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.billing.charge_usage", charge),
        patch("backend.executor.manager.billing.handle_low_balance"),
        patch("backend.executor.manager.billing.handle_insufficient_funds_notif"),
        patch("backend.executor.manager.increment_execution_count", return_value=1),
        patch("backend.executor.manager.automod_manager", automod),
        patch("backend.executor.manager.update_node_execution_status"),
        patch("backend.executor.manager.send_rate_limited_discord_alert"),
    ]
    loops = [asyncio.new_event_loop() for _ in range(2)]
    threads = [threading.Thread(target=loop.run_forever, daemon=True) for loop in loops]
    processor.node_execution_loop, processor.node_evaluation_loop = loops
    for thread in threads:
        thread.start()
    for p in patches:
        p.start()
    try:
        processor._on_graph_execution(
            graph_exec=GraphExecutionEntry(
                user_id=mid_call.user_id,
                graph_exec_id=mid_call.graph_exec_id,
                graph_id=mid_call.graph_id,
                graph_version=mid_call.graph_version,
            ),
            cancel=threading.Event(),
            log_metadata=MagicMock(),
            execution_stats=GraphExecutionStats(),
            cluster_lock=MagicMock(),
        )
    finally:
        for p in patches:
            p.stop()
        for loop, thread in zip(loops, threads):
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()
    return run


OUT_OF_CREDITS = InsufficientBalanceError(
    "Insufficient balance", user_id="user-1", balance=0, amount=PRE_FLIGHT
)
# The run's own output check refusing after the balance ended the loop
# replaces the run's error before the cleanup reads it.
OUTPUT_REFUSED = ModerationError(
    "Execution failed due to output content moderation",
    user_id="user-1",
    graph_exec_id="graph-exec-1",
    moderation_type="output",
)


@pytest.mark.parametrize(
    "output_refusal",
    [
        pytest.param(None, id="unaffordable-node"),
        pytest.param(OUTPUT_REFUSED, id="unaffordable-node-then-output-refused"),
    ],
)
@pytest.mark.parametrize(("model", "credentials"), PRICED_MODELS)
def test_a_run_its_user_ends_bills_the_requests_it_cuts_off(
    model, credentials, output_refusal
):
    """A node the balance can't pay for fails the run and cancels the nodes
    still mid-call; their requests are billed as on a stop."""
    run = _fail_mid_call(model, credentials, OUT_OF_CREDITS, output_refusal)

    assert run.statuses[-1] == ExecutionStatus.TERMINATED
    estimated_input = _estimated_input(run)
    assert estimated_input > 10_000
    estimate_cost = math.ceil(estimated_input * TOKEN_COST[model].input / 1e6)
    assert run.spent == [estimate_cost - PRE_FLIGHT]


def test_a_run_the_platform_breaks_settles_on_reported_usage():
    """The run's failure is not its user's doing, so the request it cuts off
    settles on the usage the provider reported, which is none here."""
    model, credentials = LLMModel.CLAUDE_4_5_HAIKU, anthropic_credentials

    run = _fail_mid_call(model, credentials, RuntimeError("database unreachable"))

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
