"""Tests for OrchestratorBlock per-iteration cost charging.

The OrchestratorBlock in agent mode makes multiple LLM calls in a single
node execution. The executor uses ``Block.extra_runtime_cost`` to detect
this and charge ``base_cost * (llm_call_count - 1)`` extra credits after
the block completes.
"""

import threading
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import Block
from backend.blocks.orchestrator import ExecutionParams, OrchestratorBlock
from backend.data.execution import ExecutionContext, ExecutionStatus
from backend.data.model import NodeExecutionStats
from backend.executor import billing, manager
from backend.util.exceptions import InsufficientBalanceError

# ── extra_runtime_cost hook ────────────────────────────────────────


class _NoOpBlock(Block):
    """Minimal concrete Block subclass that does not override extra_runtime_cost."""

    def __init__(self):
        super().__init__(
            id="00000000-0000-0000-0000-000000000001", description="No-op test block"
        )

    def run(self, input_data, **kwargs):  # type: ignore[override]
        yield "out", {}


class TestExtraRuntimeCost:
    """OrchestratorBlock opts into per-LLM-call billing via extra_runtime_cost."""

    def test_orchestrator_returns_nonzero_for_multiple_calls(self):
        block = OrchestratorBlock()
        stats = NodeExecutionStats(llm_call_count=3)
        assert block.extra_runtime_cost(stats) == 2

    def test_orchestrator_returns_zero_for_single_call(self):
        block = OrchestratorBlock()
        stats = NodeExecutionStats(llm_call_count=1)
        assert block.extra_runtime_cost(stats) == 0

    def test_orchestrator_returns_zero_for_zero_calls(self):
        block = OrchestratorBlock()
        stats = NodeExecutionStats(llm_call_count=0)
        assert block.extra_runtime_cost(stats) == 0

    def test_default_block_returns_zero(self):
        """A block that does not override extra_runtime_cost returns 0."""
        block = _NoOpBlock()
        stats = NodeExecutionStats(llm_call_count=10)
        assert block.extra_runtime_cost(stats) == 0


# ── charge_extra_runtime_cost math ───────────────────────────────────


@pytest.fixture()
def fake_node_exec():
    node_exec = MagicMock()
    node_exec.user_id = "u"
    node_exec.graph_exec_id = "g"
    node_exec.graph_id = "g"
    node_exec.node_exec_id = "ne"
    node_exec.node_id = "n"
    node_exec.block_id = "b"
    node_exec.inputs = {}
    return node_exec


@pytest.fixture()
def patched_processor(monkeypatch):
    """ExecutionProcessor with stubbed db client / block lookup helpers.

    Returns the processor and a list of credit amounts spent so tests can
    assert on what was charged.

    Note: ``ExecutionProcessor.__new__()`` bypasses ``__init__`` — if
    ``__init__`` gains required state in the future this fixture will need
    updating.
    """
    spent: list[int] = []

    class FakeDb:
        def spend_credits(self, *, user_id, cost, metadata):
            spent.append(cost)
            return 1000  # remaining balance

    fake_block = MagicMock()
    fake_block.name = "FakeBlock"

    monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
    monkeypatch.setattr(billing, "get_block", lambda block_id: fake_block)
    monkeypatch.setattr(
        billing,
        "block_usage_cost",
        lambda block, input_data, **_kw: (10, {"model": "claude-sonnet-4-6"}),
    )

    proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
    return proc, spent


class TestChargeExtraRuntimeCost:
    @pytest.mark.asyncio
    async def test_zero_extra_iterations_charges_nothing(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=0
        )
        assert cost == 0
        assert balance == 0
        assert spent == []

    @pytest.mark.asyncio
    async def test_extra_iterations_multiplies_base_cost(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=4
        )
        assert cost == 40  # 4 × 10
        assert balance == 1000
        assert spent == [40]

    @pytest.mark.asyncio
    async def test_negative_extra_iterations_charges_nothing(
        self, patched_processor, fake_node_exec
    ):
        proc, spent = patched_processor
        cost, balance = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=-1
        )
        assert cost == 0
        assert balance == 0
        assert spent == []

    @pytest.mark.asyncio
    async def test_capped_at_max(self, monkeypatch, fake_node_exec):
        """Runaway llm_call_count is capped at _MAX_EXTRA_RUNTIME_COST."""

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 1000

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(billing, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            billing,
            "block_usage_cost",
            lambda block, input_data, **_kw: (10, {}),
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cap = billing._MAX_EXTRA_RUNTIME_COST
        cost, _ = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=cap * 100
        )
        # Charged at most cap × 10
        assert cost == cap * 10
        assert spent == [cap * 10]

    @pytest.mark.asyncio
    async def test_zero_base_cost_skips_charge(self, monkeypatch, fake_node_exec):

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 0

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(billing, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            billing, "block_usage_cost", lambda block, input_data, **_kw: (0, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=4
        )
        assert cost == 0
        assert balance == 0
        assert spent == []

    @pytest.mark.asyncio
    async def test_block_not_found_skips_charge(self, monkeypatch, fake_node_exec):

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 0

        monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(billing, "get_block", lambda block_id: None)
        monkeypatch.setattr(
            billing, "block_usage_cost", lambda block, input_data, **_kw: (10, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = await proc.charge_extra_runtime_cost(
            fake_node_exec, extra_count=3
        )
        assert cost == 0
        assert balance == 0
        assert spent == []

    @pytest.mark.asyncio
    async def test_propagates_insufficient_balance_error(
        self, monkeypatch, fake_node_exec
    ):
        """Out-of-credits errors must propagate, not be silently swallowed."""

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                raise InsufficientBalanceError(
                    user_id=user_id,
                    message="Insufficient balance",
                    balance=0,
                    amount=cost,
                )

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(billing, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            billing, "block_usage_cost", lambda block, input_data, **_kw: (10, {})
        )

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        with pytest.raises(InsufficientBalanceError):
            await proc.charge_extra_runtime_cost(fake_node_exec, extra_count=4)


# ── charge_node_usage ──────────────────────────────────────────────


class TestChargeNodeUsage:
    """charge_node_usage delegates to billing.charge_usage with execution_count=0."""

    @pytest.mark.asyncio
    async def test_delegates_with_zero_execution_count(
        self, monkeypatch, fake_node_exec
    ):
        """Nested tool charges should NOT inflate the per-execution counter."""

        captured: dict = {}

        def fake_charge_usage(node_exec, execution_count):
            captured["execution_count"] = execution_count
            captured["node_exec"] = node_exec
            return (5, 100)

        def fake_handle_low_balance(
            db_client, user_id, current_balance, transaction_cost
        ):
            pass

        monkeypatch.setattr(billing, "charge_usage", fake_charge_usage)
        monkeypatch.setattr(billing, "handle_low_balance", fake_handle_low_balance)
        monkeypatch.setattr(billing, "get_db_client", lambda: MagicMock())

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = await proc.charge_node_usage(fake_node_exec)
        assert cost == 5
        assert balance == 100
        assert captured["execution_count"] == 0

    @pytest.mark.asyncio
    async def test_calls_handle_low_balance_when_cost_nonzero(
        self, monkeypatch, fake_node_exec
    ):
        """charge_node_usage should call handle_low_balance when total_cost > 0."""

        low_balance_calls: list[dict] = []

        def fake_charge_usage(node_exec, execution_count):
            return (10, 50)

        def fake_handle_low_balance(
            db_client, user_id, current_balance, transaction_cost
        ):
            low_balance_calls.append(
                {
                    "user_id": user_id,
                    "current_balance": current_balance,
                    "transaction_cost": transaction_cost,
                }
            )

        monkeypatch.setattr(billing, "charge_usage", fake_charge_usage)
        monkeypatch.setattr(billing, "handle_low_balance", fake_handle_low_balance)
        monkeypatch.setattr(billing, "get_db_client", lambda: MagicMock())

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = await proc.charge_node_usage(fake_node_exec)
        assert cost == 10
        assert balance == 50
        assert len(low_balance_calls) == 1
        assert low_balance_calls[0]["user_id"] == "u"
        assert low_balance_calls[0]["current_balance"] == 50
        assert low_balance_calls[0]["transaction_cost"] == 10

    @pytest.mark.asyncio
    async def test_skips_handle_low_balance_when_cost_zero(
        self, monkeypatch, fake_node_exec
    ):
        """charge_node_usage should NOT call handle_low_balance when cost is 0."""

        low_balance_calls: list = []

        def fake_charge_usage(node_exec, execution_count):
            return (0, 200)

        def fake_handle_low_balance(
            db_client, user_id, current_balance, transaction_cost
        ):
            low_balance_calls.append(True)

        monkeypatch.setattr(billing, "charge_usage", fake_charge_usage)
        monkeypatch.setattr(billing, "handle_low_balance", fake_handle_low_balance)
        monkeypatch.setattr(billing, "get_db_client", lambda: MagicMock())

        proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)
        cost, balance = await proc.charge_node_usage(fake_node_exec)
        assert cost == 0
        assert low_balance_calls == []


# ── on_node_execution charging gate ────────────────────────────────


class _FakeNode:
    """Minimal stand-in for a ``Node`` object with a block attribute."""

    def __init__(self, extra_charges: int = 0, block_name: str = "FakeBlock"):
        self.block = MagicMock()
        self.block.name = block_name
        self.block.extra_runtime_cost = MagicMock(return_value=extra_charges)


class _FakeExecContext:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run


def _make_node_exec(dry_run: bool = False) -> MagicMock:
    """Build a NodeExecutionEntry-like mock for on_node_execution tests."""
    ne = MagicMock()
    ne.user_id = "u"
    ne.graph_id = "g"
    ne.graph_exec_id = "ge"
    ne.node_id = "n"
    ne.node_exec_id = "ne"
    ne.block_id = "b"
    ne.inputs = {}
    ne.execution_context = _FakeExecContext(dry_run=dry_run)
    return ne


@pytest.fixture()
def gated_processor(monkeypatch):
    """ExecutionProcessor with on_node_execution's downstream calls stubbed.

    Lets tests flip the gate conditions (status, extra_runtime_cost result,
    llm_call_count, dry_run) and observe whether charge_extra_runtime_cost
    was called.
    """

    calls: dict[str, list] = {
        "charge_extra_runtime_cost": [],
        "handle_low_balance": [],
        "handle_insufficient_funds_notif": [],
    }

    # Stub node lookup + DB client so the wrapper doesn't touch real infra.
    fake_db = MagicMock()
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=2))
    monkeypatch.setattr(manager, "get_db_async_client", lambda: fake_db)
    monkeypatch.setattr(billing, "get_db_client", lambda: fake_db)
    # get_block is called by LogMetadata construction in on_node_execution.
    monkeypatch.setattr(
        manager,
        "get_block",
        lambda block_id: MagicMock(name="FakeBlock"),
    )
    # Persistence + cost logging are not under test here.
    monkeypatch.setattr(
        manager,
        "async_update_node_execution_status",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        manager,
        "async_update_graph_execution_state",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        manager,
        "log_system_credential_cost",
        AsyncMock(return_value=None),
    )

    proc = manager.ExecutionProcessor.__new__(manager.ExecutionProcessor)

    # Control the status returned by the inner execution call.
    inner_result = {"status": ExecutionStatus.COMPLETED, "llm_call_count": 3}

    async def fake_inner(
        self,
        *,
        node,
        node_exec,
        node_exec_progress,
        stats,
        db_client,
        log_metadata,
        nodes_input_masks=None,
        nodes_to_skip=None,
    ):
        stats.llm_call_count = inner_result["llm_call_count"]
        return MagicMock(wall_time=0.1, cpu_time=0.1), inner_result["status"]

    monkeypatch.setattr(
        manager.ExecutionProcessor,
        "_on_node_execution",
        fake_inner,
    )

    async def fake_charge_extra(node_exec, extra_count):
        calls["charge_extra_runtime_cost"].append(extra_count)
        return (extra_count * 10, 500)

    monkeypatch.setattr(billing, "charge_extra_runtime_cost", fake_charge_extra)

    def fake_low_balance(db_client, user_id, current_balance, transaction_cost):
        calls["handle_low_balance"].append(
            {
                "user_id": user_id,
                "current_balance": current_balance,
                "transaction_cost": transaction_cost,
            }
        )

    monkeypatch.setattr(billing, "handle_low_balance", fake_low_balance)

    def fake_notif(db_client, user_id, graph_id, e):
        calls["handle_insufficient_funds_notif"].append(
            {"user_id": user_id, "graph_id": graph_id, "error": e}
        )

    monkeypatch.setattr(billing, "handle_insufficient_funds_notif", fake_notif)

    return proc, calls, inner_result, fake_db, NodeExecutionStats


@pytest.mark.asyncio
async def test_on_node_execution_charges_extra_iterations_when_gate_passes(
    gated_processor,
):
    """COMPLETED + extra_runtime_cost > 0 + not dry_run → charged."""

    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.COMPLETED
    inner["llm_call_count"] = 3  # → extra_charges = 2
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=2))

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    assert calls["charge_extra_runtime_cost"] == [2]
    # handle_low_balance must be called with the remaining balance returned by
    # charge_extra_runtime_cost (500) so users are alerted when balance drops low.
    assert len(calls["handle_low_balance"]) == 1


@pytest.mark.asyncio
async def test_on_node_execution_skips_when_status_not_completed(gated_processor):

    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.FAILED
    inner["llm_call_count"] = 5
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=4))

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    assert calls["charge_extra_runtime_cost"] == []


@pytest.mark.asyncio
async def test_on_node_execution_skips_when_extra_charges_zero(gated_processor):

    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.COMPLETED
    inner["llm_call_count"] = 5
    # Block returns 0 extra charges (base class default)
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=0))

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    assert calls["charge_extra_runtime_cost"] == []


@pytest.mark.asyncio
async def test_on_node_execution_skips_when_dry_run(gated_processor):

    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.COMPLETED
    inner["llm_call_count"] = 5
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=4))

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=True),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    assert calls["charge_extra_runtime_cost"] == []


@pytest.mark.asyncio
async def test_on_node_execution_insufficient_balance_records_error_and_notifies(
    monkeypatch,
    gated_processor,
):
    """When extra-iteration charging fails with InsufficientBalanceError:

    - the run still reports COMPLETED (the work is already done)
    - execution_stats.error is NOT set (would flip node_error_count and
      leak balance amounts into persisted node_stats — see manager.py
      comment in the IBE handler)
    - _handle_insufficient_funds_notif is called so the user is notified
    - the structured ERROR log is the alerting hook
    """

    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.COMPLETED
    inner["llm_call_count"] = 4
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=3))

    async def raise_ibe(node_exec, extra_count):
        raise InsufficientBalanceError(
            user_id=node_exec.user_id,
            message="Insufficient balance",
            balance=0,
            amount=extra_count * 10,
        )

    monkeypatch.setattr(billing, "charge_extra_runtime_cost", raise_ibe)

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    result_stats = await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    # error stays None — node ran to completion, only the post-hoc
    # charge failed. Setting .error would (a) flip node_error_count++
    # creating an "errored COMPLETED node" inconsistency, and (b) leak
    # balance amounts into persisted node_stats.
    assert result_stats.error is None
    # User notification fired.
    assert len(calls["handle_insufficient_funds_notif"]) == 1
    assert calls["handle_insufficient_funds_notif"][0]["user_id"] == "u"


# ── Orchestrator _execute_single_tool_with_manager charging gates ──


async def _run_tool_exec_with_stats(
    *,
    dry_run: bool,
    tool_stats_error,
    charge_node_usage_mock=None,
):
    """Invoke _execute_single_tool_with_manager against fully mocked deps
    and return (charge_call_count, merge_stats_calls).

    Used to prove the dry_run and error guards around charge_node_usage
    behave as documented, and that InsufficientBalanceError propagates.
    """
    block = OrchestratorBlock()

    # Mocked async DB client used inside orchestrator.
    mock_db_client = AsyncMock()
    mock_target_node = MagicMock()
    mock_target_node.block_id = "test-block-id"
    mock_target_node.input_default = {}
    mock_db_client.get_node.return_value = mock_target_node
    mock_node_exec_result = MagicMock()
    mock_node_exec_result.node_exec_id = "test-tool-exec-id"
    mock_db_client.upsert_execution_input.return_value = (
        mock_node_exec_result,
        {"query": "t"},
    )
    mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {"result": "ok"}

    # ExecutionProcessor mock: on_node_execution returns supplied error.
    mock_processor = AsyncMock()
    mock_processor.running_node_execution = defaultdict(MagicMock)
    mock_processor.execution_stats = MagicMock()
    mock_processor.execution_stats_lock = threading.Lock()
    mock_node_stats = MagicMock()
    mock_node_stats.error = tool_stats_error
    mock_processor.on_node_execution = AsyncMock(return_value=mock_node_stats)
    mock_processor.charge_node_usage = charge_node_usage_mock or AsyncMock(
        return_value=(10, 990)
    )

    # Build a tool_info shaped like _build_tool_info_from_args output.
    tool_call = MagicMock()
    tool_call.id = "call-1"
    tool_call.name = "search_keywords"
    tool_call.arguments = '{"query":"t"}'
    tool_def = {
        "type": "function",
        "function": {
            "name": "search_keywords",
            "_sink_node_id": "test-sink-node-id",
            "_field_mapping": {},
            "parameters": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    tool_info = OrchestratorBlock._build_tool_info_from_args(
        tool_call_id="call-1",
        tool_name="search_keywords",
        tool_args={"query": "t"},
        tool_def=tool_def,
    )

    exec_params = ExecutionParams(
        user_id="u",
        graph_id="g",
        node_id="n",
        graph_version=1,
        graph_exec_id="ge",
        node_exec_id="ne",
        execution_context=ExecutionContext(
            human_in_the_loop_safe_mode=False, dry_run=dry_run
        ),
    )

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db_client,
    ):
        try:
            await block._execute_single_tool_with_manager(
                tool_info, exec_params, mock_processor, responses_api=False
            )
            raised = None
        except Exception as e:
            raised = e

    return mock_processor.charge_node_usage, raised


@pytest.mark.asyncio
async def test_tool_execution_skips_charging_on_dry_run():
    """dry_run=True → charge_node_usage is NOT called."""
    charge_mock, raised = await _run_tool_exec_with_stats(
        dry_run=True, tool_stats_error=None
    )
    assert raised is None
    assert charge_mock.call_count == 0


@pytest.mark.asyncio
async def test_tool_execution_skips_charging_on_failed_tool():
    """tool_node_stats.error is an Exception → charge_node_usage NOT called."""
    charge_mock, raised = await _run_tool_exec_with_stats(
        dry_run=False, tool_stats_error=RuntimeError("tool blew up")
    )
    assert raised is None
    assert charge_mock.call_count == 0


@pytest.mark.asyncio
async def test_tool_execution_skips_charging_on_cancelled_tool():
    """Cancellation (BaseException subclass) → charge_node_usage NOT called.

    Guards the fix for sentry's BaseException concern: the old
    `isinstance(error, Exception)` check would have treated CancelledError
    as "no error" and billed the user for a terminated run.
    """
    import asyncio as _asyncio

    charge_mock, raised = await _run_tool_exec_with_stats(
        dry_run=False, tool_stats_error=_asyncio.CancelledError()
    )
    assert raised is None
    assert charge_mock.call_count == 0


@pytest.mark.asyncio
async def test_tool_execution_insufficient_balance_propagates():
    """InsufficientBalanceError from charge_node_usage must propagate out.

    If this leaked into a ToolCallResult the LLM loop would keep running
    with 'tool failed' errors and the user would get unpaid work.
    """
    raising_charge = AsyncMock(
        side_effect=InsufficientBalanceError(
            user_id="u", message="nope", balance=0, amount=10
        )
    )
    _, raised = await _run_tool_exec_with_stats(
        dry_run=False,
        tool_stats_error=None,
        charge_node_usage_mock=raising_charge,
    )
    assert isinstance(raised, InsufficientBalanceError)


@pytest.mark.asyncio
async def test_tool_execution_on_node_execution_returns_none_sets_is_error():
    """on_node_execution returning None (swallowed by @async_error_logged) must
    result in a tool response with _is_error=True so the LLM loop knows the
    tool failed and does not treat a silent error as a successful execution.
    """
    block = OrchestratorBlock()

    mock_db_client = AsyncMock()
    mock_target_node = MagicMock()
    mock_target_node.block_id = "test-block-id"
    mock_target_node.input_default = {}
    mock_db_client.get_node.return_value = mock_target_node
    mock_node_exec_result = MagicMock()
    mock_node_exec_result.node_exec_id = "test-tool-exec-id"
    mock_db_client.upsert_execution_input.return_value = (
        mock_node_exec_result,
        {"query": "t"},
    )

    mock_processor = AsyncMock()
    mock_processor.running_node_execution = defaultdict(MagicMock)
    mock_processor.execution_stats = MagicMock()
    mock_processor.execution_stats_lock = threading.Lock()
    # on_node_execution returns None — simulates @async_error_logged(swallow=True)
    # swallowing an internal error
    mock_processor.on_node_execution = AsyncMock(return_value=None)

    tool_call = MagicMock()
    tool_call.id = "call-none"
    tool_call.name = "search_keywords"
    tool_call.arguments = '{"query":"t"}'
    tool_def = {
        "type": "function",
        "function": {
            "name": "search_keywords",
            "_sink_node_id": "test-sink-node-id",
            "_field_mapping": {},
            "parameters": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    tool_info = OrchestratorBlock._build_tool_info_from_args(
        tool_call_id="call-none",
        tool_name="search_keywords",
        tool_args={"query": "t"},
        tool_def=tool_def,
    )

    exec_params = ExecutionParams(
        user_id="u",
        graph_id="g",
        node_id="n",
        graph_version=1,
        graph_exec_id="ge",
        node_exec_id="ne",
        execution_context=ExecutionContext(
            human_in_the_loop_safe_mode=False, dry_run=False
        ),
    )

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db_client,
    ):
        resp = await block._execute_single_tool_with_manager(
            tool_info, exec_params, mock_processor, responses_api=False
        )

    assert resp.get("_is_error") is True
    # charge_node_usage must NOT be called for a failed tool execution
    mock_processor.charge_node_usage.assert_not_called()


# ── on_node_execution FAILED + InsufficientBalanceError notification ──


@pytest.mark.asyncio
async def test_on_node_execution_failed_ibe_sends_notification(
    monkeypatch,
    gated_processor,
):
    """When status == FAILED and execution_stats.error is InsufficientBalanceError,
    _handle_insufficient_funds_notif must be called.

    This path fires when a nested tool charge inside the orchestrator raises
    InsufficientBalanceError, which propagates out of the block's run() generator
    and is caught by _on_node_execution's broad except, setting status=FAILED and
    execution_stats.error=IBE. on_node_execution's post-execution block then
    sends the user notification so they understand why the run stopped.
    """

    proc, calls, inner, fake_db, NodeExecutionStats = gated_processor
    ibe = InsufficientBalanceError(
        user_id="u",
        message="Insufficient balance",
        balance=0,
        amount=30,
    )

    # Simulate _on_node_execution returning FAILED with IBE in stats.error.
    async def fake_inner_failed(
        self,
        *,
        node,
        node_exec,
        node_exec_progress,
        stats,
        db_client,
        log_metadata,
        nodes_input_masks=None,
        nodes_to_skip=None,
    ):
        stats.error = ibe
        return MagicMock(wall_time=0.1, cpu_time=0.1), ExecutionStatus.FAILED

    monkeypatch.setattr(
        manager.ExecutionProcessor,
        "_on_node_execution",
        fake_inner_failed,
    )
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=0))

    stats_pair = (
        MagicMock(
            node_count=0, nodes_cputime=0, nodes_walltime=0, cost=0, node_error_count=0
        ),
        threading.Lock(),
    )
    await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    # The notification must have fired so the user knows why their run stopped.
    assert len(calls["handle_insufficient_funds_notif"]) == 1
    assert calls["handle_insufficient_funds_notif"][0]["user_id"] == "u"
    # charge_extra_runtime_cost must NOT be called — status is FAILED.
    assert calls["charge_extra_runtime_cost"] == []


# ── Billing leak: non-IBE exception during extra-iteration charging ──


@pytest.mark.asyncio
async def test_on_node_execution_non_ibe_billing_failure_keeps_completed(
    monkeypatch,
    gated_processor,
):
    """When charge_extra_runtime_cost raises a non-IBE exception (e.g. DB outage):

    - execution_stats.error stays None (node ran to completion)
    - status stays COMPLETED (work already done)
    - the billing_leak error is logged but does not corrupt execution_stats
    """
    proc, calls, inner, fake_db, _ = gated_processor
    inner["status"] = ExecutionStatus.COMPLETED
    inner["llm_call_count"] = 4
    fake_db.get_node = AsyncMock(return_value=_FakeNode(extra_charges=3))

    async def raise_conn_error(node_exec, extra_count):
        raise ConnectionError("DB connection lost")

    monkeypatch.setattr(billing, "charge_extra_runtime_cost", raise_conn_error)

    stats_pair = (
        MagicMock(
            node_count=0,
            nodes_cputime=0,
            nodes_walltime=0,
            cost=0,
            node_error_count=0,
        ),
        threading.Lock(),
    )
    result_stats = await proc.on_node_execution(
        node_exec=_make_node_exec(dry_run=False),
        node_exec_progress=MagicMock(),
        nodes_input_masks=None,
        graph_stats_pair=stats_pair,
    )
    # error stays None — node completed, only billing failed.
    assert result_stats.error is None
    # No notification was sent (only IBE triggers notification).
    assert len(calls["handle_insufficient_funds_notif"]) == 0


# ── _charge_usage with execution_count=0 ──


class TestChargeUsageZeroExecutionCount:
    """Verify _charge_usage(node_exec, 0) does not invoke execution_usage_cost."""

    def test_execution_count_zero_skips_execution_tier(self, monkeypatch):
        """_charge_usage with execution_count=0 must not call execution_usage_cost."""
        execution_tier_called = []

        def fake_execution_usage_cost(count):
            execution_tier_called.append(count)
            return (100, count)

        spent: list[int] = []

        class FakeDb:
            def spend_credits(self, *, user_id, cost, metadata):
                spent.append(cost)
                return 500

        fake_block = MagicMock()
        fake_block.name = "FakeBlock"

        monkeypatch.setattr(billing, "get_db_client", lambda: FakeDb())
        monkeypatch.setattr(billing, "get_block", lambda block_id: fake_block)
        monkeypatch.setattr(
            billing,
            "block_usage_cost",
            lambda block, input_data, **_kw: (10, {}),
        )
        monkeypatch.setattr(billing, "execution_usage_cost", fake_execution_usage_cost)

        ne = MagicMock()
        ne.user_id = "u"
        ne.graph_exec_id = "ge"
        ne.graph_id = "g"
        ne.node_exec_id = "ne"
        ne.node_id = "n"
        ne.block_id = "b"
        ne.inputs = {}

        total_cost, remaining = billing.charge_usage(ne, 0)
        assert total_cost == 10  # block cost only
        assert remaining == 500
        assert spent == [10]
        # execution_usage_cost must NOT have been called
        assert execution_tier_called == []
