import asyncio
import logging
import random
from collections.abc import Awaitable
from typing import Any, cast

from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.helpers import validate_group_ids
from graphiti_core.utils.datetime_utils import convert_datetimes_to_strings

from .config import graphiti_config

logger = logging.getLogger(__name__)

# graphiti-core's ``FalkorDriver.execute_query`` logs terminal query errors
# under this logger name. Our ``execute_query`` override no longer delegates to
# ``super()`` (so it can keep intermediate retries silent), so it re-emits
# terminal errors under the SAME logger to preserve Sentry grouping — and to
# keep the SENTRY-1387 benign-teardown filter, which keys on this logger,
# working on the query path.
_UPSTREAM_QUERY_LOGGER = logging.getLogger("graphiti_core.driver.falkordb_driver")

# FalkorDB sheds load with this message when its global pending-query queue is
# full. It is a *pre-execution* rejection — the query never ran — so retrying
# is side-effect-free and safe even for writes.
_PENDING_QUEUE_OVERFLOW = "max pending queries exceeded"

# Cap each retry's total wait (jitter included) so a mis-tuned high
# ``falkordb_query_max_attempts`` / ``falkordb_query_backoff_base`` can't balloon
# one wait into minutes. At default settings total backoff stays well within the
# warm-context timeout budget.
_MAX_RETRY_DELAY_SECONDS = 2.0


def _is_pending_queue_overflow(exc: Exception) -> bool:
    return _PENDING_QUEUE_OVERFLOW in str(exc).lower()


class AutoGPTFalkorDriver(FalkorDriver):
    """FalkorDriver subclass with three AutoGPT-specific tweaks.

    1. ``build_fulltext_query`` adds the per-user ``group_id`` filter so
       multi-tenant searches don't cross user graphs.

    2. ``build_indices`` parameter (defaults True for upstream-compatible
       behaviour) opts out of the fire-and-forget
       ``build_indices_and_constraints`` background task that
       graphiti-core's ``FalkorDriver.__init__`` always spawns.
       That task is fine for long-lived drivers (chat ingest path) but
       generates "Connection closed by server" / "Buffer is closed" log
       spam when the driver is created per short-lived request — most
       notably the admin memory visualizer's per-request driver opens,
       where the indexing task's sequential CREATE INDEX statements
       race the route's own queries and the closing of the connection
       when the route returns. Pass ``build_indices=False`` for
       read-only paths against an existing user's graph; the indices
       are already there from the long-lived chat-write client.

    3. ``execute_query`` retries FalkorDB's transient "Max pending queries
       exceeded" backpressure with bounded jittered backoff, so a load
       spike degrades into a slightly slower memory op instead of a
       dropped one plus a Sentry alert. (SENTRY-1384.)
    """

    def __init__(self, *args, build_indices: bool = True, **kwargs):
        # Stash the flag BEFORE super().__init__ runs because
        # FalkorDriver.__init__ fires
        # ``loop.create_task(self.build_indices_and_constraints())``
        # synchronously; our override below reads this attribute when
        # the task actually ticks on the loop.
        self._build_indices_at_init = build_indices
        super().__init__(*args, **kwargs)

    async def build_indices_and_constraints(self) -> None:  # type: ignore[override]
        if not getattr(self, "_build_indices_at_init", True):
            # Caller asserted indices already exist (or will be built by
            # someone else) — skip the multi-CREATE-INDEX race that
            # produces the log spam.
            return
        await super().build_indices_and_constraints()

    async def execute_query(
        self, cypher_query_, **kwargs
    ) -> tuple[list[dict[str, Any]], list[str], None] | None:
        """Run a Cypher query, retrying transient pending-queue overflow.

        "Max pending queries exceeded" is FalkorDB backpressure: the shared
        server's pending-query queue is full under concurrent memory traffic
        (ingest fan-out, warm-context searches, community rebuilds). It clears
        within milliseconds, so the rejected query is retried with jittered
        exponential backoff. Only an exhausted retry budget reaches Sentry —
        intermediate attempts stay silent. Every other error (Cypher typo,
        missing graph, connection teardown) fails fast on the first attempt,
        exactly as upstream ``FalkorDriver.execute_query``.

        Reimplemented rather than delegating to ``super()`` so per-attempt
        logging is under our control; the result-shaping mirrors upstream.
        """
        graph = self._get_graph(self._database)
        # falkordb's async ``Graph.query`` is typed as its sync counterpart in
        # the stubs (returns a bare ``QueryResult``, params typed narrowly);
        # cast to the real awaitable/param types rather than suppress. Same
        # runtime call upstream ``FalkorDriver.execute_query`` makes.
        params = cast(dict[str, object], convert_datetimes_to_strings(dict(kwargs)))
        attempts = max(1, graphiti_config.falkordb_query_max_attempts)

        for attempt in range(attempts):
            try:
                result = await cast(Awaitable[Any], graph.query(cypher_query_, params))
                return self._to_records(result)
            except Exception as e:
                if "already indexed" in str(e):
                    _UPSTREAM_QUERY_LOGGER.info(f"Index already exists: {e}")
                    return None
                if _is_pending_queue_overflow(e) and attempt < attempts - 1:
                    delay = self._pending_queue_retry_delay(attempt)
                    logger.debug(
                        "FalkorDB pending-queue overflow; retry %s/%s in %.2fs",
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # ``params`` hold user memory content (names, facts) — omit them
                # from this Sentry-routed log to avoid leaking PII. The exception
                # and query text are enough to triage a genuine failure.
                _UPSTREAM_QUERY_LOGGER.error(
                    f"Error executing FalkorDB query: {e}\n{cypher_query_}"
                )
                raise
        raise AssertionError("unreachable: loop returns or raises on every path")

    @staticmethod
    def _pending_queue_retry_delay(attempt: int) -> float:
        """Jittered exponential backoff (capped) so retries drain the queue over
        time and don't synchronize across concurrent callers."""
        base = graphiti_config.falkordb_query_backoff_base
        delay = base * (2**attempt) + random.uniform(0, base)
        return min(delay, _MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _to_records(result) -> tuple[list[dict[str, Any]], list[str], None]:
        """Mirror of upstream's list-of-lists → list-of-dicts result shaping."""
        header = [h[1] for h in result.header]
        records = [
            {name: (row[i] if i < len(row) else None) for i, name in enumerate(header)}
            for row in result.result_set
        ]
        return records, header, None

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 128,
    ) -> str:
        validate_group_ids(group_ids)

        group_filter = ""
        if group_ids:
            group_filter = f"(@group_id:{'|'.join(group_ids)})"

        sanitized_query = self.sanitize(query)
        query_words = sanitized_query.split()
        filtered_words = [word for word in query_words if word.lower() not in STOPWORDS]
        sanitized_query = " | ".join(filtered_words)

        if not sanitized_query:
            fulltext_query = group_filter
        elif not group_filter:
            fulltext_query = f"({sanitized_query})"
        else:
            fulltext_query = f"{group_filter} ({sanitized_query})"

        if len(fulltext_query) >= max_query_length:
            return ""

        return fulltext_query
