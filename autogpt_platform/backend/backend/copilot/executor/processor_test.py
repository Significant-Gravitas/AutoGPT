"""Unit tests for CoPilot mode routing logic in the processor.

Tests cover the mode→service mapping:
  - 'fast' → baseline service
  - 'extended_thinking' → SDK service
  - None → feature flag / config fallback

as well as the ``CHAT_MODE_OPTION`` server-side gate.  The tests import
the real production helpers from ``processor.py`` so the routing logic
has meaningful coverage.
"""

import asyncio
import concurrent.futures
import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.executor.processor import (
    CoPilotProcessor,
    resolve_effective_mode,
    resolve_use_sdk_for_mode,
    sync_fail_close_session,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata


class TestResolveUseSdkForMode:
    """Tests for the per-request mode routing logic."""

    @pytest.mark.asyncio
    async def test_fast_mode_uses_baseline(self):
        """mode='fast' always routes to baseline, regardless of flags."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    "fast",
                    "user-1",
                    use_claude_code_subscription=True,
                    config_default=True,
                )
                is False
            )

    @pytest.mark.asyncio
    async def test_extended_thinking_uses_sdk(self):
        """mode='extended_thinking' always routes to SDK, regardless of flags."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    "extended_thinking",
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_uses_subscription_override(self):
        """mode=None with claude_code_subscription=True routes to SDK."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=True,
                    config_default=False,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_uses_feature_flag(self):
        """mode=None with feature flag enabled routes to SDK."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ) as flag_mock:
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is True
            )
            flag_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_none_mode_uses_config_default(self):
        """mode=None falls back to config.use_claude_agent_sdk."""
        # When LaunchDarkly returns the default (True), we expect SDK routing.
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=True,
                )
                is True
            )

    @pytest.mark.asyncio
    async def test_none_mode_all_disabled(self):
        """mode=None with all flags off routes to baseline."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert (
                await resolve_use_sdk_for_mode(
                    None,
                    "user-1",
                    use_claude_code_subscription=False,
                    config_default=False,
                )
                is False
            )


class TestResolveEffectiveMode:
    """Tests for the CHAT_MODE_OPTION server-side gate."""

    @pytest.mark.asyncio
    async def test_none_mode_passes_through(self):
        """mode=None is returned as-is without a flag check."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ) as flag_mock:
            assert await resolve_effective_mode(None, "user-1") is None
            flag_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mode_stripped_when_flag_disabled(self):
        """When CHAT_MODE_OPTION is off, mode is dropped to None."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ):
            assert await resolve_effective_mode("fast", "user-1") is None
            assert await resolve_effective_mode("extended_thinking", "user-1") is None

    @pytest.mark.asyncio
    async def test_mode_preserved_when_flag_enabled(self):
        """When CHAT_MODE_OPTION is on, the user-selected mode is preserved."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ):
            assert await resolve_effective_mode("fast", "user-1") == "fast"
            assert (
                await resolve_effective_mode("extended_thinking", "user-1")
                == "extended_thinking"
            )

    @pytest.mark.asyncio
    async def test_anonymous_user_with_mode(self):
        """Anonymous users (user_id=None) still pass through the gate."""
        with patch(
            "backend.copilot.executor.processor.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ) as flag_mock:
            assert await resolve_effective_mode("fast", None) is None
            flag_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# _execute_async aclose propagation
# ---------------------------------------------------------------------------


class _TrackedStream:
    """Minimal async-generator stand-in that records whether ``aclose``
    was called, so tests can verify the processor forces explicit cleanup
    of the published stream on every exit path (normal + break on cancel)."""

    def __init__(self, events: list):
        self._events = events
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def aclose(self) -> None:
        self.aclose_called = True


def _make_entry() -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="sess-1",
        turn_id="turn-1",
        user_id="user-1",
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
    )


def _make_log() -> CoPilotLogMetadata:
    return CoPilotLogMetadata(logger=logging.getLogger("test-copilot"))


class TestExecuteAsyncAclose:
    """``_execute_async`` must call ``aclose`` on the published stream both
    when the loop exits naturally and when ``cancel`` is set mid-stream —
    otherwise ``stream_chat_completion_sdk`` stays suspended and keeps
    holding the per-session Redis lock until GC."""

    def _patches(self, published_stream: _TrackedStream):
        """Shared mock context: patches every dependency ``_execute_async``
        touches so the aclose path is the only behaviour under test."""
        return [
            patch(
                "backend.copilot.executor.processor.ChatConfig",
                return_value=MagicMock(test_mode=True, use_claude_agent_sdk=True),
            ),
            patch(
                "backend.copilot.executor.processor.stream_chat_completion_dummy",
                return_value=MagicMock(),
            ),
            patch(
                "backend.copilot.executor.processor.stream_registry.stream_and_publish",
                return_value=published_stream,
            ),
            patch(
                "backend.copilot.executor.processor.stream_registry.mark_session_completed",
                new=AsyncMock(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_normal_exit_calls_aclose(self) -> None:
        published = _TrackedStream(events=[MagicMock(), MagicMock()])
        proc = CoPilotProcessor()
        cancel = threading.Event()
        cluster_lock = MagicMock()

        patches = self._patches(published)
        with patches[0], patches[1], patches[2], patches[3]:
            await proc._execute_async(_make_entry(), cancel, cluster_lock, _make_log())

        assert published.aclose_called is True

    @pytest.mark.asyncio
    async def test_cancel_break_calls_aclose(self) -> None:
        events = [MagicMock()]  # first chunk delivered, then cancel fires
        published = _TrackedStream(events=events)
        proc = CoPilotProcessor()
        cancel = threading.Event()
        cancel.set()  # pre-set so the loop breaks on the first chunk
        cluster_lock = MagicMock()

        patches = self._patches(published)
        with patches[0], patches[1], patches[2], patches[3]:
            await proc._execute_async(_make_entry(), cancel, cluster_lock, _make_log())

        assert published.aclose_called is True


@pytest.fixture
def exec_loop():
    """Long-lived asyncio loop on a daemon thread — mirrors the layout
    ``CoPilotProcessor`` sets up (``execution_loop`` + ``execution_thread``)
    so ``sync_fail_close_session`` has a real cross-thread loop to submit
    into via ``run_coroutine_threadsafe``."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


class TestSyncFailCloseSession:
    """``sync_fail_close_session`` is the last-line-of-defense invoked from
    ``CoPilotProcessor.execute``'s ``finally``. It must call
    ``mark_session_completed`` via the processor's long-lived
    ``execution_loop`` (cross-thread submit) and must swallow Redis
    failures so a transient outage doesn't propagate out of the finally."""

    def test_invokes_mark_session_completed_with_shutdown_message(
        self, exec_loop
    ) -> None:
        mock_mark = AsyncMock()
        with patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mock_mark,
        ):
            sync_fail_close_session("sess-1", _make_log(), exec_loop)

        mock_mark.assert_awaited_once()
        assert mock_mark.await_args is not None
        assert mock_mark.await_args.args[0] == "sess-1"
        assert "shut down" in mock_mark.await_args.kwargs["error_message"].lower()

    def test_swallows_redis_error(self, exec_loop) -> None:
        # Raising from the mock ensures the helper catches the exception
        # instead of propagating it back into execute()'s finally block.
        mock_mark = AsyncMock(side_effect=RuntimeError("redis down"))
        with patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mock_mark,
        ):
            sync_fail_close_session("sess-2", _make_log(), exec_loop)  # must not raise

        mock_mark.assert_awaited_once()

    def test_closed_execution_loop_skipped_cleanly(self) -> None:
        """If cleanup_worker has already stopped the execution_loop by the
        time the safety net fires, ``run_coroutine_threadsafe`` raises
        RuntimeError. Expected behavior: log + return without propagating."""
        dead_loop = asyncio.new_event_loop()
        dead_loop.close()

        mock_mark = AsyncMock()
        with patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mock_mark,
        ):
            # Must not raise even though the loop is closed
            sync_fail_close_session("sess-closed-loop", _make_log(), dead_loop)

        # mark_session_completed was never scheduled because the loop was dead
        mock_mark.assert_not_awaited()

    def test_bounded_timeout_when_redis_hangs(self, exec_loop) -> None:
        """Scenario D: Redis unreachable — the inner ``asyncio.wait_for``
        must fire and the helper must return without blocking the worker.

        Simulates a wedged Redis by sleeping past the 10s fail-close budget.
        The helper must return within the configured grace (+ a small
        scheduler margin) and must not re-raise.
        """
        import time as _time

        from backend.copilot.executor.processor import _FAIL_CLOSE_REDIS_TIMEOUT

        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(_FAIL_CLOSE_REDIS_TIMEOUT + 5)

        with patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=_hang,
        ):
            start = _time.monotonic()
            sync_fail_close_session(
                "sess-hang", _make_log(), exec_loop
            )  # must not raise
            elapsed = _time.monotonic() - start

        # wait_for fires at _FAIL_CLOSE_REDIS_TIMEOUT; outer future.result
        # has +2s slack. If the timeout is missing/broken the helper would
        # block the full sleep duration (~15s).
        assert elapsed < _FAIL_CLOSE_REDIS_TIMEOUT + 4.0, (
            f"sync_fail_close_session hung for {elapsed:.1f}s — bounded "
            f"timeout did not fire"
        )


# ---------------------------------------------------------------------------
# End-to-end execute() safety-net coverage — the PR's core invariant
# ---------------------------------------------------------------------------


class TestExecuteSafetyNet:
    """``CoPilotProcessor.execute`` must always invoke
    ``sync_fail_close_session`` in its ``finally`` so a session never stays
    ``status=running`` in Redis.

    Validates the four deploy-time scenarios the PR targets:

    * A — SIGTERM mid-turn: ``cancel`` event fires, ``_execute`` returns,
      safety net still runs.
    * B — happy path: normal completion, safety net runs (cheap CAS no-op).
    * C — zombie Redis state: the async ``mark_session_completed`` in
      ``_execute_async`` blows up, but the outer safety net marks the
      session failed anyway.
    * D — covered by ``TestSyncFailCloseSession::test_bounded_timeout…``.
    """

    def _attach_exec_loop(self, proc: CoPilotProcessor, loop) -> None:
        """``execute`` dispatches the safety net onto ``self.execution_loop``.
        Tests don't call ``on_executor_start`` (which spawns the real
        per-worker loop), so wire the shared fixture loop in directly."""
        proc.execution_loop = loop

    def _run_execute_in_thread(self, proc: CoPilotProcessor, cancel: threading.Event):
        """``CoPilotProcessor.execute`` expects to be called from a pool
        worker thread that has *no* running event loop, so we always run
        it off the main thread to preserve that invariant. Returns the
        future so callers can inspect both result and exception paths."""
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(proc.execute, _make_entry(), cancel, MagicMock())
            # Block until execute() returns (or raises) so the safety net
            # has run by the time we inspect mocks.
            try:
                fut.result(timeout=30)
            except BaseException:
                pass
            return fut
        finally:
            pool.shutdown(wait=True)

    def test_happy_path_invokes_safety_net(self, exec_loop) -> None:
        """Scenario B: normal completion still runs the sync safety net.
        Proves the ``finally`` always fires, even when nothing went wrong —
        ``mark_session_completed``'s atomic CAS makes this a cheap no-op
        in production."""
        mock_mark = AsyncMock()
        proc = CoPilotProcessor()
        self._attach_exec_loop(proc, exec_loop)
        with patch.object(proc, "_execute"), patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mock_mark,
        ):
            self._run_execute_in_thread(proc, threading.Event())

        mock_mark.assert_awaited_once()
        assert mock_mark.await_args is not None
        assert mock_mark.await_args.args[0] == "sess-1"

    def test_sigterm_mid_turn_invokes_safety_net(self, exec_loop) -> None:
        """Scenario A: worker raises (simulating future.cancel + grace
        timeout escaping ``_execute``); ``execute`` must still reach the
        safety net in its ``finally`` and mark the session failed."""
        mock_mark = AsyncMock()
        proc = CoPilotProcessor()
        self._attach_exec_loop(proc, exec_loop)
        with patch.object(
            proc,
            "_execute",
            side_effect=concurrent.futures.TimeoutError("grace expired"),
        ), patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mock_mark,
        ):
            self._run_execute_in_thread(proc, threading.Event())

        mock_mark.assert_awaited_once()

    def test_zombie_redis_async_path_still_marks_session_failed(
        self, exec_loop
    ) -> None:
        """Scenario C: ``_execute_async``'s own ``mark_session_completed``
        call is broken (simulating the exact async-Redis hiccup that caused
        the original zombie sessions). The outer ``sync_fail_close_session``
        runs on the processor's long-lived ``execution_loop`` and succeeds
        where the async path failed."""
        call_log: list[str] = []

        async def _ok(*args, **kwargs):
            call_log.append("sync-ok")

        def _broken_execute(entry, cancel, cluster_lock, log):
            # Simulate the async path raising because its Redis client is
            # wedged (the pre-fix zombie-session scenario).
            raise RuntimeError("async Redis client broken")

        proc = CoPilotProcessor()
        self._attach_exec_loop(proc, exec_loop)
        with patch.object(proc, "_execute", side_effect=_broken_execute), patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=_ok,
        ):
            self._run_execute_in_thread(proc, threading.Event())

        # The sync safety net must have fired despite the async path
        # blowing up — this is the core guarantee of the PR.
        assert call_log == [
            "sync-ok"
        ], f"expected sync_fail_close_session to run once, got {call_log!r}"

    def test_cancel_waits_for_async_task_to_finish(self, exec_loop) -> None:
        """A cancel request must not let ``_execute`` return while the
        underlying asyncio task is still cleaning up. Returning early would
        make the manager release the session lock while late stream writes
        are still possible."""
        proc = CoPilotProcessor()
        self._attach_exec_loop(proc, exec_loop)

        started = threading.Event()
        cancel_seen = threading.Event()
        release_cleanup = threading.Event()
        finished = threading.Event()

        async def _stubborn_cancel(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancel_seen.set()
                while not release_cleanup.is_set():
                    await asyncio.sleep(0.01)
            finally:
                finished.set()

        proc._execute_async = _stubborn_cancel  # type: ignore[method-assign]

        cancel = threading.Event()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(
                proc._execute,
                _make_entry(),
                cancel,
                MagicMock(),
                _make_log(),
            )
            assert started.wait(timeout=5)

            cancel.set()
            assert cancel_seen.wait(timeout=5)
            assert not fut.done()

            release_cleanup.set()
            fut.result(timeout=5)
            assert finished.is_set()
        finally:
            pool.shutdown(wait=True)

    def test_cancel_wait_has_bounded_escape_hatch(self, exec_loop) -> None:
        """A wedged async cleanup must not keep the worker refreshing the
        session lock forever; after the grace window, ``_execute`` returns
        so ``execute`` can run the sync fail-close safety net."""
        proc = CoPilotProcessor()
        self._attach_exec_loop(proc, exec_loop)

        started = threading.Event()
        cancel_seen = threading.Event()
        release_cleanup = threading.Event()
        finished = threading.Event()

        async def _wedged_cancel(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancel_seen.set()
                while not release_cleanup.is_set():
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        pass
            finally:
                finished.set()

        proc._execute_async = _wedged_cancel  # type: ignore[method-assign]

        cancel = threading.Event()
        cluster_lock = MagicMock()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            with patch(
                "backend.copilot.executor.processor._CANCEL_GRACE_SECONDS",
                0.05,
            ):
                fut = pool.submit(
                    proc._execute,
                    _make_entry(),
                    cancel,
                    cluster_lock,
                    _make_log(),
                )
                assert started.wait(timeout=5)

                cancel.set()
                assert cancel_seen.wait(timeout=5)
                fut.result(timeout=5)

            assert not finished.is_set()
            assert cluster_lock.refresh.call_count < 10

            release_cleanup.set()
            assert finished.wait(timeout=5)
        finally:
            pool.shutdown(wait=True)
