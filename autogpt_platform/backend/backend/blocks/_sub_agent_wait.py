"""How AgentExecutorBlock waits on its child run.

The child's events stream in over Redis pub/sub, which keeps no history, so the
stream is only the fast path. Whenever it may have missed events, or goes quiet,
the database says what happened: outputs come oldest first, each once, and the
wait ends when the child has, however its end became known.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from backend.blocks._base import BlockType
from backend.data.db_accessors import execution_db
from backend.data.execution import (
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
    GraphExecutionEvent,
    GraphExecutionMeta,
    NodeExecutionEvent,
    NodeExecutionResult,
)

if TYPE_CHECKING:
    from backend.executor.utils import LogMetadata

_logger = logging.getLogger(__name__)

TERMINAL_STATUSES = (
    ExecutionStatus.COMPLETED,
    ExecutionStatus.TERMINATED,
    ExecutionStatus.FAILED,
)
# A child's end can go unheard on a healthy stream too (its executor failed to
# publish it), so after this long without an event the database is asked.
# Bounds how long such a miss holds the parent run, which has no timeout.
QUIET_RECHECK_S = 60.0
# How soon a catch-up the stream still owes is retried after it failed; the
# delay doubles with each failure in a row, up to QUIET_RECHECK_S. Until then,
# events other than the child's end wait for that retry.
CATCH_UP_RETRY_S = 5.0
# Failed catch-ups after the child's end was seen live before the wait ends on
# that end anyway: outputs missed in a gap may then be missing.
CATCH_UP_TRIES_AFTER_END = 3
# Outputs held in memory while a catch-up is owed. Past this, a chatty child
# in a long outage would hold every payload; the catch-up's read returns the
# rest, in order, and only a wait that gives up on the database loses them.
MAX_HELD_OUTPUTS = 100

Output = tuple[str, Any]


class SubAgentRunGoneError(Exception):
    """The child's run is gone from the database, not just deleted from the
    user's history, so nothing will ever finish it."""


class SubAgentWait:
    """One wait on one child run. Iterate ``outputs()``; once it ends, the
    child has finished and ``stats`` holds its totals."""

    def __init__(
        self,
        events: AsyncIterator[ExecutionEvent | None],
        *,
        user_id: str,
        graph_exec_id: str,
        log_id: str,
        logger: "LogMetadata",
    ) -> None:
        self._events = events  # listen_with_resync: None marks a gap
        self._user_id = user_id
        self._graph_exec_id = graph_exec_id
        self._log_id = log_id
        self._logger = logger
        self._queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue()
        self._yielded: set[str] = set()
        # Events may have been missed and not caught up yet.
        self._stale = False
        # The child's end, seen live while a catch-up was owed: it ends the
        # wait once that catch-up succeeds, as the child publishes no more.
        self._pending_end: GraphExecutionEvent | None = None
        self._failures = 0  # failed catch-ups in a row
        # When an owed catch-up is retried. Absolute, so the child's
        # non-terminal updates arriving meanwhile don't keep pushing it out.
        self._retry_at: float | None = None
        # Outputs seen live while a catch-up was owed, yielded after it.
        self._held: list[NodeExecutionEvent] = []
        self._failures_after_end = 0
        self.finished = False
        self.stats: GraphExecutionMeta.Stats | None = None

    async def outputs(self) -> AsyncGenerator[Output, None]:
        reader = asyncio.create_task(self._read())
        try:
            while not self.finished:
                for output in await self._step(await self._next()):
                    yield output
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    async def _read(self) -> None:
        try:
            async for event in self._events:
                self._queue.put_nowait(event)
        except Exception:
            _logger.exception(
                f"{self._log_id}: event stream failed; checking the database "
                f"every {QUIET_RECHECK_S:.0f}s instead"
            )

    async def _next(self) -> ExecutionEvent | None:
        """The next event; ``None`` for a gap, a quiet spell or a retry."""
        wait = QUIET_RECHECK_S
        if self._retry_at is not None:
            wait = self._retry_at - asyncio.get_running_loop().time()
            if wait <= 0:
                # Due: retry before reading on. A queued event wouldn't wait
                # for the timeout below, so the retry would trail them all.
                self._retry_at = None
                return None
        try:
            async with asyncio.timeout(wait):
                event = await self._queue.get()
        except TimeoutError:
            # A quiet spell, or a scheduled retry fell due: catch up now.
            self._retry_at = None
            return None
        self._stale = self._stale or event is None
        return event

    async def _step(self, event: ExecutionEvent | None) -> list[Output]:
        if event is not None:
            if event.status not in TERMINAL_STATUSES:
                self._logger.info(
                    f"Execution {self._log_id} skipping event {event.event_type} "
                    f"status={event.status} "
                    f"node={getattr(event, 'node_exec_id', '?')}"
                )
                return []
            if not self._stale:
                if event.event_type == ExecutionEventType.NODE_EXEC_UPDATE:
                    return self._outputs_of(event)
                self._finish(event.stats, f"graph completed: {event.status}")
                return []
            if event.event_type == ExecutionEventType.GRAPH_EXEC_UPDATE:
                # The child's last event, so one read at most: try at once.
                self._pending_end = event
                return await self._try_catch_up()
            self._hold(event)
        if self._retry_at is not None:
            # A failed catch-up's retry is scheduled and reads this too, so a
            # busy child adds no read per finished node to a failing database.
            return []
        # A gap, a quiet spell, or an event while events may have been missed
        # (which must not overtake them): the database has the rest, in order.
        return await self._try_catch_up()

    async def _try_catch_up(self) -> list[Output]:
        try:
            outputs, finished = await self._catch_up()
        except SubAgentRunGoneError:
            if self._pending_end is None:
                raise
            end = self._pending_end
            self._finish(end.stats, f"graph completed: {end.status}; run since gone")
            return self._release_held()
        except Exception:
            self._on_failed_catch_up()
            # It finishes the wait when it gives up on the database after the
            # child's end was seen; what was seen live still comes out then.
            return self._release_held() if self.finished else []
        self._stale, self._failures, self._retry_at = False, 0, None
        # In case the database hadn't recorded them yet.
        outputs += self._release_held()
        if finished is not None:
            self._finish(finished.stats, f"found {finished.status} in the database")
        elif self._pending_end is not None:
            # The database hadn't recorded the end yet.
            end = self._pending_end
            self._finish(end.stats, f"graph completed: {end.status}")
        return outputs

    async def _catch_up(self) -> tuple[list[Output], GraphExecutionMeta | None]:
        """Outputs the stream may have missed, oldest first, and the child's
        execution if it has finished."""
        meta = await self._read_run()
        finished = meta if meta.status in TERMINAL_STATUSES else None
        if not (self._stale or finished):
            return [], None
        from backend.blocks import get_output_block_ids

        # Read with the user's id just above, so these nodes are the user's.
        nodes = await execution_db().get_node_executions(
            graph_exec_id=self._graph_exec_id,
            statuses=list(TERMINAL_STATUSES),
            block_ids=list(get_output_block_ids()),
        )
        # Reversed first: equal times then keep the order they were queued in.
        nodes = sorted(reversed(nodes), key=lambda node: node.end_time or node.add_time)
        outputs = [output for node in nodes for output in self._outputs_of(node)]
        return outputs, finished

    async def _read_run(self) -> GraphExecutionMeta:
        """The child's run, including one deleted from the user's history:
        deleting a run doesn't stop it, so it may still be running and
        publishing. Only a row that's truly absent raises."""
        db = execution_db()
        meta = await db.get_graph_execution_meta(
            user_id=self._user_id, execution_id=self._graph_exec_id
        )
        if meta is not None:
            return meta
        deleted = await db.get_graph_executions(
            graph_exec_id=self._graph_exec_id, user_id=self._user_id, deleted=True
        )
        if not deleted:
            self._logger.warning(f"Execution {self._log_id}: the run no longer exists")
            # Shown as the node's error: workflow wording, no internal ids.
            raise SubAgentRunGoneError(
                "This sub-workflow's run no longer exists, so it can't finish."
            )
        return deleted[0]

    def _hold(self, event: NodeExecutionEvent) -> None:
        from backend.blocks import get_output_block_ids

        if len(self._held) < MAX_HELD_OUTPUTS and (
            event.block_id in get_output_block_ids()
        ):
            self._held.append(event)

    def _release_held(self) -> list[Output]:
        outputs = [output for event in self._held for output in self._outputs_of(event)]
        self._held.clear()
        return outputs

    def _outputs_of(self, node: NodeExecutionResult) -> list[Output]:
        from backend.blocks import get_block

        log_id, logger = self._log_id, self._logger
        if node.node_exec_id in self._yielded:
            logger.debug(f"{log_id} already yielded node execution {node.node_exec_id}")
            return []
        self._yielded.add(node.node_exec_id)

        if not node.block_id:
            logger.warning(f"{log_id} received event without block_id {node}")
            return []

        block = get_block(node.block_id)
        if not block or block.block_type != BlockType.OUTPUT:
            return []

        output_name = node.input_data.get("name")
        if not output_name:
            logger.warning(f"{log_id} produced an output with no name {node}")
            return []

        logger.debug(f"Execution {log_id} produced {output_name}")
        return [(output_name, data) for data in node.output_data.get("output", [])]

    def _on_failed_catch_up(self) -> None:
        """Still owed: retried with backoff, and nothing newer is yielded
        meanwhile. With the child's end already seen, give up after a few
        tries rather than wait on the database for ever."""
        self._logger.warning(f"Execution {self._log_id} catch-up failed", exc_info=True)
        self._stale = True
        self._failures += 1
        backoff = CATCH_UP_RETRY_S * 2 ** (self._failures - 1)
        self._retry_at = asyncio.get_running_loop().time() + min(
            backoff, QUIET_RECHECK_S
        )
        if self._pending_end is None:
            return
        self._failures_after_end += 1
        if self._failures_after_end >= CATCH_UP_TRIES_AFTER_END:
            end = self._pending_end
            self._logger.warning(
                f"Execution {self._log_id} ended without its last catch-up; "
                "outputs from a stream gap may be missing"
            )
            self._finish(end.stats, f"graph completed: {end.status}")

    def _finish(self, stats: GraphExecutionMeta.Stats | None, how: str) -> None:
        self._logger.info(
            f"Execution {self._log_id} {how}, yielded {len(self._yielded)} outputs"
        )
        self.finished, self.stats = True, stats
