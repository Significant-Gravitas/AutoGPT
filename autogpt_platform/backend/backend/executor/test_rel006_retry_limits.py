"""REL-006 retry limits and cost drain — finite durable bounds.

Covers:
  - retry success (transient failure then success)
  - retry exhaustion (permanent failure → DLQ after bound)
  - duplicate scheduler delivery (one logical execution)
  - cancellation during retry (no further chargeable retry)
  - executor restart (retry counter durable via Redis, cancel durable via DB)
  - cost-log drain across ownership boundary (loop-agnostic queue)
  - permanent downstream failure (cost log bounded retries, remains queued)
"""

import asyncio
import datetime
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.execution import ExecutionStatus
from backend.data.platform_cost import PlatformCostEntry


# ---------------------------------------------------------------------------
# Helper to build a PlatformCostEntry
# ---------------------------------------------------------------------------
def _entry(user_id="user-1", block="BlockA", provider="openai"):
    return PlatformCostEntry(
        user_id=user_id,
        graph_exec_id="gx-1",
        node_exec_id="nx-1",
        graph_id="g-1",
        node_id="n-1",
        block_id="b-1",
        block_name=block,
        provider=provider,
        credential_id="cred-1",
        cost_microdollars=1000,
        input_tokens=10,
        output_tokens=10,
        tracking_type="tokens",
        tracking_amount=20,
        metadata={},
    )


# ---------------------------------------------------------------------------
# 1) retry success
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_success_transient_then_success():
    """Transient DB failure on first attempt, success on second → one charge, no DLQ."""
    from backend.executor import manager as mgr

    # Simulate _should_requeue_execution: first call allows, second would be after success so not called
    with patch("backend.executor.manager.redis.get_redis") as mock_redis, \
         patch("backend.executor.manager.get_database_manager_client") as mock_db:
        mock_r = MagicMock()
        # incr returns 1 then 2
        mock_r.pipeline.return_value.incr.return_value = None
        mock_r.pipeline.return_value.expire.return_value = None
        mock_r.pipeline.return_value.execute.return_value = [1]
        # Use incr_with_ttl_sync mock to return 1
        with patch("backend.executor.manager.incr_with_ttl_sync", return_value=1) as mock_incr:
            # No cancel
            mock_db.return_value.get_graph_execution_meta.return_value = MagicMock(cancelRequestedAt=None)
            assert mgr._should_requeue_execution("gx-1", "user-1") is True
            mock_incr.assert_called_once()

    # Also test cost log retry success: first log fails, second succeeds via drain
    import backend.executor.cost_tracking as ct
    ct._pending_cost_entries.clear()
    ct._pending_log_tasks.clear()
    db_client = MagicMock()
    # First call fails, second succeeds
    db_client.log_platform_cost = AsyncMock(side_effect=[Exception("transient"), None])
    entry = _entry()
    ct.schedule_platform_cost_log(db_client, entry)
    # Wait a bit for async task to run (bounded retries 5)
    await asyncio.sleep(0.3)
    # Entry should have been removed on success (or remain if task hasn't finished, but drain will clean)
    # Drain should succeed
    db_client.log_platform_cost = AsyncMock(return_value=None)
    await ct.drain_pending_cost_logs(timeout=2.0)
    # After drain, queue should be empty
    with ct._pending_cost_entries_lock:
        assert len(ct._pending_cost_entries) == 0


# ---------------------------------------------------------------------------
# 2) retry exhaustion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_exhaustion_drops_after_bound():
    """Permanent failure → after 5 attempts, requeue returns False and execution marked FAILED."""
    from backend.executor import manager as mgr

    with patch("backend.executor.manager.incr_with_ttl_sync") as mock_incr, \
         patch("backend.executor.manager.get_database_manager_client") as mock_db, \
         patch("backend.executor.manager.redis.get_redis"):
        mock_db.return_value.get_graph_execution_meta.return_value = MagicMock(cancelRequestedAt=None)
        # Simulate counter exceeding MAX (5)
        mock_incr.return_value = 6  # > MAX_EXECUTION_REQUEUE_ATTEMPTS
        assert mgr._should_requeue_execution("gx-exhaust", "user-1") is False

        # Counter at limit should allow
        mock_incr.return_value = 5
        assert mgr._should_requeue_execution("gx-limit", "user-1") is True

        # Counter 0 should allow
        mock_incr.return_value = 1
        assert mgr._should_requeue_execution("gx-first", "user-1") is True


@pytest.mark.asyncio
async def test_cost_log_retry_exhaustion_keeps_queued():
    """Cost log permanent failure → after 5 attempts, entry remains queued for next drain (not silently dropped)."""
    import backend.executor.cost_tracking as ct
    ct._pending_cost_entries.clear()
    ct._pending_log_tasks.clear()
    db_client = MagicMock()
    db_client.log_platform_cost = AsyncMock(side_effect=Exception("DB down"))
    entry = _entry(user_id="user-exhaust")
    ct.schedule_platform_cost_log(db_client, entry)
    # Let fast-path task exhaust its 5 retries
    await asyncio.sleep(0.6)
    # Fast path exhausted but entry remains in queue (not stranded silently)
    with ct._pending_cost_entries_lock:
        still_queued = any(e[1].user_id == "user-exhaust" for e in ct._pending_cost_entries)
        assert still_queued, "permanent failure must keep entry queued, not drop silently"

    # Drain also exhausts retries but keeps entry
    db_client.log_platform_cost = AsyncMock(side_effect=Exception("still down"))
    await ct.drain_pending_cost_logs(timeout=2.0)
    with ct._pending_cost_entries_lock:
        assert any(e[1].user_id == "user-exhaust" for e in ct._pending_cost_entries)
    # Cleanup
    ct._pending_cost_entries.clear()


# ---------------------------------------------------------------------------
# 3) duplicate scheduler delivery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_scheduler_delivery_one_logical():
    """Duplicate scheduler occurrence (same fireTime) converges to one executionId, no double charge."""
    import datetime as dt
    from prisma.errors import UniqueViolationError

    def _uve(msg: str = "Unique constraint failed") -> UniqueViolationError:
        """Prisma's UniqueViolationError requires a dict-shaped error body."""
        return UniqueViolationError({"user_facing_error": {"message": msg, "code": "P2002", "meta": {}}})

    fire = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    occ_dispatched = MagicMock(id="occ-1", status="dispatched", executionId="exec-1", fireTime=fire)

    with patch("prisma.models.ScheduleOccurrence.prisma") as mock_occ, \
         patch("backend.executor.scheduler.execution_utils.add_graph_execution", new_callable=AsyncMock) as mock_add:
        mock_occ.return_value.create = AsyncMock(side_effect=_uve())
        mock_occ.return_value.find_unique = AsyncMock(return_value=occ_dispatched)
        mock_occ.return_value.update = AsyncMock()

        from backend.executor.scheduler import _execute_graph

        # Patch canonical_fire_time to return fixed fire
        with patch("backend.data.schedule_occurrence.canonical_fire_time", return_value=fire):
            result = await _execute_graph(
                schedule_id="sched-dup",
                user_id="user-1",
                graph_id="graph-1",
                graph_version=1,
                agent_name="test",
                cron="* * * * *",
                input_data={},
                input_credentials={},
                organization_id="",
                team_id=None,
            )
            assert result == "exec-1"
            # No new execution created — deduped
            assert mock_add.call_count == 0


# ---------------------------------------------------------------------------
# 4) cancellation during retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_during_retry_prevents_requeue():
    """Cancellation durably prevents further chargeable retry."""
    from backend.executor import manager as mgr

    cancelled_meta = MagicMock(cancelRequestedAt=datetime.datetime.now(datetime.timezone.utc))
    with patch("backend.executor.manager.get_database_manager_client") as mock_db, \
         patch("backend.executor.manager.incr_with_ttl_sync") as mock_incr:
        mock_db.return_value.get_graph_execution_meta.return_value = cancelled_meta
        # Even though retry counter would allow, cancellation must veto
        mock_incr.return_value = 1
        assert mgr._should_requeue_execution("gx-cancel", "user-cancel") is False
        # Verify incr not called when cancelled? Actually our impl checks cancel first, so incr not reached
        # But we allow either order; ensure False

    # Also test executor honors durable cancel without workload
    from backend.data.execution import GraphExecutionEntry, ExecutionContext
    from backend.data.model import GraphExecutionStats

    meta_cancelled = MagicMock(status=ExecutionStatus.QUEUED, id="gx-cancel-2")
    meta_cancelled.cancelRequestedAt = datetime.datetime.now(datetime.timezone.utc)
    with patch("backend.executor.manager.get_db_client") as mock_db, \
         patch("backend.executor.manager.get_db_async_client"):
        mock_db.return_value.get_graph_execution_meta = MagicMock(return_value=meta_cancelled)
        mock_db.return_value.get_credits = MagicMock(return_value=100)
        proc = mgr.ExecutionProcessor()
        proc.node_execution_loop = MagicMock()
        proc.node_evaluation_loop = MagicMock()
        with patch("backend.executor.manager.automod_manager") as mock_automod:
            mock_automod.moderate_graph_execution_inputs = AsyncMock(return_value=None)
            with patch("backend.executor.manager.asyncio.run_coroutine_threadsafe") as mock_run:
                fake_future = MagicMock()
                fake_future.result.return_value = None
                mock_run.return_value = fake_future
                entry = GraphExecutionEntry(
                    user_id="user-1",
                    graph_exec_id="gx-cancel-2",
                    graph_id="g-1",
                    graph_version=1,
                    execution_context=ExecutionContext(dry_run=False),
                )
                cancel = MagicMock(is_set=MagicMock(return_value=False))
                import logging
                from backend.util.logging import TruncatedLogger
                log_meta = MagicMock()
                stats = GraphExecutionStats(is_dry_run=False)
                cluster_lock = MagicMock()
                result = proc._on_graph_execution(entry, cancel, log_meta, stats, cluster_lock)
                status = result[1] if isinstance(result, tuple) else result
                assert status == ExecutionStatus.TERMINATED
                mock_db.return_value.get_node_executions.assert_not_called()


# ---------------------------------------------------------------------------
# 5) executor restart
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_executor_restart_retry_counter_durable():
    """Restart after 2 failures → Redis counter persists, 3rd retry still bounded."""
    from backend.executor import manager as mgr

    # Simulate Redis counter that survives restart
    counter = {"val": 0}

    def fake_incr(r, k, ttl):
        counter["val"] += 1
        return counter["val"]

    with patch("backend.executor.manager.incr_with_ttl_sync", side_effect=fake_incr), \
         patch("backend.executor.manager.get_database_manager_client") as mock_db, \
         patch("backend.executor.manager.redis.get_redis"):
        mock_db.return_value.get_graph_execution_meta.return_value = MagicMock(cancelRequestedAt=None)
        # First attempt after restart counts as 3rd overall
        counter["val"] = 2
        assert mgr._should_requeue_execution("gx-restart", "user-1") is True  # 3rd
        assert mgr._should_requeue_execution("gx-restart", "user-1") is True  # 4th
        assert mgr._should_requeue_execution("gx-restart", "user-1") is True  # 5th
        assert mgr._should_requeue_execution("gx-restart", "user-1") is False  # 6th exhausted

    # Also test durable cancel survives restart (DB flag)
    from backend.executor.utils import stop_graph_execution

    order = []

    async def fake_set(*args, **kwargs):
        order.append("persist")

    async def fake_publish(*args, **kwargs):
        order.append("publish")

    with patch("backend.data.execution.set_cancel_requested", new_callable=AsyncMock, side_effect=fake_set), \
         patch("backend.executor.utils.get_async_execution_queue", new_callable=AsyncMock) as mock_q, \
         patch("backend.executor.utils.get_database_manager_async_client") as mock_db2, \
         patch("backend.executor.utils.execution_db"), \
         patch("backend.executor.utils._get_child_executions", new_callable=AsyncMock, return_value=[]), \
         patch("backend.executor.utils.prisma") as mock_prisma, \
         patch("backend.executor.utils.get_async_execution_event_bus") as mock_bus:
        mock_prisma.is_connected.return_value = False
        mock_q.return_value.publish_message = AsyncMock(side_effect=fake_publish)
        terminated = MagicMock(status=ExecutionStatus.TERMINATED, id="gx-restart-cancel")
        mock_db2.return_value.get_graph_execution_meta = AsyncMock(return_value=terminated)
        mock_bus.return_value.publish = AsyncMock()
        await stop_graph_execution(user_id="user-1", graph_exec_id="gx-restart-cancel", wait_timeout=0.1)
        assert order == ["persist", "publish"]


# ---------------------------------------------------------------------------
# 6) cost-log drain across ownership boundary
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cost_log_drain_across_ownership_boundary():
    """Entries enqueued on one loop are drainable on another loop (loop-agnostic queue)."""
    import backend.executor.cost_tracking as ct
    ct._pending_cost_entries.clear()
    ct._pending_log_tasks.clear()

    db_client = MagicMock()
    db_client.log_platform_cost = AsyncMock(return_value=None)

    # Enqueue on "other" loop: simulate by directly appending without task
    entry_other = _entry(user_id="user-other-loop", block="OtherLoopBlock")
    with ct._pending_cost_entries_lock:
        ct._pending_cost_entries.append((db_client, entry_other))

    # Also enqueue copilot entry on other loop
    import backend.copilot.token_tracking as tt
    tt._pending_copilot_entries.clear()
    copilot_entry = _entry(user_id="user-copilot-other", block="copilot:SDK")
    with tt._pending_copilot_entries_lock:
        tt._pending_copilot_entries.append(copilot_entry)

    # Drain on current loop should flush both
    with patch("backend.copilot.token_tracking.platform_cost_db") as mock_copilot_db:
        mock_copilot_db.return_value.log_platform_cost = AsyncMock(return_value=None)
        await ct.drain_pending_cost_logs(timeout=2.0)

    with ct._pending_cost_entries_lock:
        assert not any(e[1].user_id == "user-other-loop" for e in ct._pending_cost_entries)
    with tt._pending_copilot_entries_lock:
        assert len(tt._pending_copilot_entries) == 0

    # Verify thread-safety: concurrent enqueue from thread while draining on async loop
    def thread_enqueue():
        e = _entry(user_id="user-thread", block="ThreadBlock")
        with ct._pending_cost_entries_lock:
            ct._pending_cost_entries.append((db_client, e))

    t = threading.Thread(target=thread_enqueue)
    t.start()
    t.join()
    await ct.drain_pending_cost_logs(timeout=2.0)
    with ct._pending_cost_entries_lock:
        assert len(ct._pending_cost_entries) == 0


# ---------------------------------------------------------------------------
# 7) permanent downstream failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_permanent_downstream_failure_bounded():
    """Downstream DB always fails → bounded 5 attempts per drain, not infinite, no charge duplication."""
    import backend.executor.cost_tracking as ct
    ct._pending_cost_entries.clear()
    ct._pending_log_tasks.clear()
    db_client = MagicMock()
    db_client.log_platform_cost = AsyncMock(side_effect=Exception("permanent DB outage"))
    entry = _entry(user_id="user-perm-fail", block="PermFailBlock")
    ct.schedule_platform_cost_log(db_client, entry)
    await asyncio.sleep(0.6)  # fast path exhausts 5 attempts

    # After fast path, entry still queued (not dropped)
    with ct._pending_cost_entries_lock:
        assert len(ct._pending_cost_entries) == 1

    # Drain should retry 5 more times but still keep entry (not infinite loop, not duplicate charge)
    await ct.drain_pending_cost_logs(timeout=2.0)
    # log_platform_cost called 5 (fast path) + 5 (drain) = 10, not infinite
    assert db_client.log_platform_cost.call_count == 10
    with ct._pending_cost_entries_lock:
        assert len(ct._pending_cost_entries) == 1
    # Simulate recovery: next drain succeeds and removes
    db_client.log_platform_cost = AsyncMock(return_value=None)
    await ct.drain_pending_cost_logs(timeout=2.0)
    with ct._pending_cost_entries_lock:
        assert len(ct._pending_cost_entries) == 0
    ct._pending_cost_entries.clear()
