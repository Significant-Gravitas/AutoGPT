import asyncio
import logging
import random
import re
from collections.abc import Awaitable
from typing import Any, cast

from graphiti_core.driver.driver import GraphDriver
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


# Cypher clauses that mutate the graph. Deliberately CONSERVATIVE: anything
# matching takes the write path, so a false positive merely preserves today's
# behaviour, while a false negative would send a real write to RO_QUERY and
# fail it. The runtime fallback below covers that case anyway.
_WRITE_CLAUSE_RE = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|FOREACH)\b", re.IGNORECASE
)

# FalkorDB's error when the graph key does not exist yet.
_EMPTY_KEY_ERROR = "invalid graph operation on empty key"

# FalkorDB's error when a write is attempted through GRAPH.RO_QUERY:
# "graph.RO_QUERY is to be executed only on read-only queries".
#
# Match the full phrase, not a bare "read-only" substring. A spurious match
# would send a genuine READ down the write path and materialize the graph —
# the exact bug this routing exists to prevent. Redis's own replica error says
# "read only" with no hyphen, so it cannot collide with this.
_RO_VIOLATION = "read-only quer"


# Quoted string literals, and // or /* */ comments. Stripped before clause
# detection so a value like ``{kind: 'CREATE'}`` in an otherwise read-only
# query is not mistaken for a write clause — that misread would send the query
# down the graph-materializing path and reintroduce the very bug this fixes.
_CYPHER_NOISE_RE = re.compile(
    r"'(?:[^'\\]|\\.)*'"  # single-quoted literal
    r"|\"(?:[^\"\\]|\\.)*\""  # double-quoted literal
    r"|`(?:[^`]|``)*`"  # back-quoted identifier
    r"|//[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment
    re.DOTALL,
)


# Procedure calls that WRITE but contain no standalone clause keyword, so
# _WRITE_CLAUSE_RE alone misses them: `\bCREATE\b` does not match
# `createNodeIndex` — there is no word boundary between "create" and "Node".
# graphiti-core builds its three fulltext indices this way
# (graphiti_core/graph_queries.py), so every one of them was classified
# read-only and bounced off RO_QUERY on the first write for every new group.
_WRITE_PROCEDURE_RE = re.compile(
    r"\bdb\.idx\.fulltext\.(createNodeIndex|createRelationshipIndex|drop)\b"
    r"|\bdb\.create\.",
    re.IGNORECASE,
)


def _strip_cypher_noise(cypher_query_: str) -> str:
    """Blank out literals/comments so only real clause keywords remain."""
    return _CYPHER_NOISE_RE.sub(" ", cypher_query_ or "")


def _is_read_only_cypher(cypher_query_: str) -> bool:
    """True when the query contains no mutating clause.

    Read-only queries go to ``GRAPH.RO_QUERY``. This matters far more than
    performance: ``GRAPH.QUERY`` MATERIALIZES THE GRAPH even for a pure MATCH,
    so routing reads through it silently creates an empty, permanently
    resident graph for every user merely looked at. That is what filled
    FalkorDB to its maxmemory ceiling twice (2026-08-16 and again 2026-08-23,
    the latter in a single weekly community-rebuild sweep: 91 -> 19,033
    graphs, 100% of sampled ones empty).
    """
    text = _strip_cypher_noise(cypher_query_)
    return not (_WRITE_CLAUSE_RE.search(text) or _WRITE_PROCEDURE_RE.search(text))


def _is_pending_queue_overflow(exc: Exception) -> bool:
    return _PENDING_QUEUE_OVERFLOW in str(exc).lower()


class AutoGPTFalkorDriver(FalkorDriver):
    """FalkorDriver subclass with three AutoGPT-specific tweaks.

    1. ``build_fulltext_query`` adds the per-user ``group_id`` filter so
       multi-tenant searches don't cross user graphs.

    2. ``build_indices`` parameter (defaults **False**) opts out of the
       fire-and-forget ``build_indices_and_constraints`` background task
       that graphiti-core's ``FalkorDriver.__init__`` always spawns.

       ``CREATE INDEX`` is a *write*, and in FalkorDB a write
       **materializes the graph**. So letting that task run on every
       driver construction mints a fully-indexed, permanently-resident
       graph for any user we merely *look at* — a search that returns
       nothing, a warm-context read, a community-rebuild sweep, even the
       debugging cookbook in this package's AGENTS.md. In prod that
       produced 13,732 graphs of which ~99.7% held zero nodes, and their
       index scaffolding (~750KB accounted each) is what pinned FalkorDB
       at ``maxmemory`` and made it reject every write for 13 days.

       The default is therefore False: constructing a driver must never
       be able to create a graph. Paths that genuinely write call
       ``ensure_indices()`` explicitly, which is safe because the graph
       is about to exist anyway.

    3. ``execute_query`` retries FalkorDB's transient "Max pending queries
       exceeded" backpressure with bounded jittered backoff, so a load
       spike degrades into a slightly slower memory op instead of a
       dropped one plus a Sentry alert. (SENTRY-1384.)
    """

    def __init__(self, *args, build_indices: bool = False, **kwargs):
        # Stash the flag BEFORE super().__init__ runs because
        # FalkorDriver.__init__ fires
        # ``loop.create_task(self.build_indices_and_constraints())``
        # synchronously; our override below reads this attribute when
        # the task actually ticks on the loop.
        self._build_indices_at_init = build_indices
        super().__init__(*args, **kwargs)

    async def build_indices_and_constraints(self) -> None:  # type: ignore[override]
        if not self._build_indices_at_init:
            # Default path. Suppresses graphiti-core's init-time task so a
            # bare driver construction cannot materialize an empty graph.
            return
        await super().build_indices_and_constraints()

    async def ensure_indices(self) -> None:
        """Build indices regardless of the ``build_indices`` init flag.

        For write paths only. Bypasses the ``__init__`` suppression above
        because the caller is about to write — the graph will exist either
        way, so the indices cost nothing extra and searches over it need
        them. Callers should invoke this once per graph, not per write.
        """
        await super().build_indices_and_constraints()

    def clone(self, database: str) -> "GraphDriver":
        """Clone onto the SUBCLASS, never a plain upstream ``FalkorDriver``.

        Upstream's ``clone()`` constructs a bare ``FalkorDriver``, which would
        re-enable the init-time ``build_indices`` task (#14052) *and* route
        every read back through ``graph.query`` — resurrecting both mass
        graph-materialization bugs with no test covering it.

        Not exercised today: every AutoGPT call site passes a single group_id.
        graphiti-core clones only when ``group_id != driver._database`` or when
        a search spans 2+ groups, so this is one line of insurance against a
        future multi-group read.
        """
        if database == self._database:
            return self
        # Upstream also special-cases ``default_group_id`` ('\\_') and maps it to
        # a 'default_db' database. That branch is unreachable here: both
        # graphiti.py clone sites call validate_group_id() first, whose
        # ^[a-zA-Z0-9_-]+$ pattern rejects the backslash, and the group_id=None
        # path never clones. Noted rather than mirrored — if it ever did become
        # reachable, this would create a graph literally named '\\_'.
        return AutoGPTFalkorDriver(
            falkor_db=self.client, database=database, build_indices=False
        )

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
        read_only = _is_read_only_cypher(cypher_query_)

        attempt = 0
        while attempt < attempts:
            try:
                return self._to_records(
                    await self._dispatch(graph, cypher_query_, params, read_only)
                )
            except Exception as e:
                message = str(e).lower()
                if read_only and _EMPTY_KEY_ERROR in message:
                    # No graph for this group yet, which simply means no
                    # memories. Return an empty result rather than raising —
                    # and deliberately do NOT retry via ``graph.query``, since
                    # that would materialize the graph and reintroduce the bug
                    # this routing exists to prevent.
                    #
                    # CAVEAT: FalkorDB checks key existence BEFORE
                    # read-only-ness, so a MISCLASSIFIED WRITE against a
                    # missing graph lands here rather than in the RO-violation
                    # branch below, and is silently dropped. The classifier
                    # covers every write in graphiti-core and this repo today
                    # (clause keywords + write procedures), but log the query
                    # so a future novel write clause is traceable rather than
                    # invisible. Debug level because a genuine read against a
                    # group with no memories yet is entirely normal and common.
                    logger.debug(
                        "Read on a group with no graph yet; returning empty. "
                        "Query: %s",
                        cypher_query_,
                    )
                    return [], [], None
                if read_only and _RO_VIOLATION in message:
                    # The classifier called a write read-only. Degrade to the
                    # write path and log the query so the missing clause or
                    # procedure can be added to _WRITE_CLAUSE_RE /
                    # _WRITE_PROCEDURE_RE.
                    #
                    # `read_only = False` + `continue` retries on the SAME
                    # attempt: the counter is not advanced here, so the write
                    # actually runs. Consuming an attempt instead would, on the
                    # final attempt or with max_attempts=1, skip the write
                    # entirely and fall through to the "unreachable" assertion.
                    logger.warning(
                        "Query classified read-only but rejected by RO_QUERY; "
                        "retrying as a write. Query: %s",
                        cypher_query_,
                    )
                    read_only = False
                    continue
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
                    attempt += 1
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
    async def _dispatch(graph, cypher_query_, params, read_only: bool) -> Any:
        """Issue one attempt on the read-only or writing transport.

        ``GRAPH.RO_QUERY`` cannot materialize a graph; ``GRAPH.QUERY`` does,
        even for a bare MATCH. Keeping the choice here means every caller goes
        through one place.
        """
        if read_only:
            return await cast(Awaitable[Any], graph.ro_query(cypher_query_, params))
        return await cast(Awaitable[Any], graph.query(cypher_query_, params))

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
