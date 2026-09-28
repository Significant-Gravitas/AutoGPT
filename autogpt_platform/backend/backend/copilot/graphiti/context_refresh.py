"""Follow-up warm context: the refresh on every later user turn (SECRT-2378).

The first turn of a session reads memory once (``context.fetch_warm_context``);
without a refresh, every later turn's recall depends on the model choosing
to call ``memory_search``, which it often skips. Every later user turn calls
``refresh_warm_context`` with that turn's message. It fetches when the
message carries at least ``WARM_CONTEXT_REFRESH_MIN_WORDS`` signal units
(``should_refresh_warm_context``), or when the SDK engine forces it because
it compacted the history for the query it is sending; otherwise it returns
``None`` without touching the graph. A fetch is one retrieval through
``context.fetch_warm_context``: the first turn's search methods reranked with
reciprocal rank fusion (one query embedding, no LLM call), the recent-episode
read and the recall policy's last check, recording no ratification hits. So
it reads memory exactly as the first turn does, and a fact forgotten between
two turns is not in the next turn's refresh.

The engines start it as a task (``start_refresh``) and, once the turn's
query is ready, wait at most ``warm_context_refresh_join_grace_ms`` for it
(``join_refresh``): that is the most it adds to time-to-first-token. The SDK
engine starts it before the query build, so the retrieval overlaps
compaction and the rest of the query's preparation; the baseline engine,
the retries and a refresh forced by a compaction start it at the join, so
the grace is their whole budget. A refresh still running at the end of the
grace is cancelled and logged at INFO ("refresh late, skipped", with how
long it had run), which is the data the grace is tuned from; one that fails
or passes ``context_refresh_timeout`` returns ``None``; either way the turn
goes ahead without a block. The engines decide which turns are follow-ups
and where the block goes: ``sdk/service.py``
(``_start_follow_up_warm_context``, ``_append_follow_up_warm_context``,
``_resend_with_fresh_warm_context``) and ``baseline/service.py``
(``_refresh_follow_up_warm_context``).
"""

import asyncio
import logging

from graphiti_core.search.search_config import EdgeReranker, SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER
from pydantic import BaseModel, ConfigDict

from .config import graphiti_config
from .context import fetch_warm_context

logger = logging.getLogger(__name__)

# Minimum "signal unit" count for a follow-up user message to trigger a
# warm-context refresh.  Short acknowledgements ("ok", "thanks", "yes") carry
# no new retrieval signal, so refreshing on them would waste a graph search +
# embedding call every turn.  A post-compaction turn bypasses this gate via
# ``refresh_warm_context(..., force=True)``.
#
# Three, not four: the failures this exists to fix are task STARTS mid-session
# ("restart the executor", "deploy prod now", "resume the migration" — all
# exactly three units), so a four-unit floor would exclude the very case it
# targets.  Three lets a stray acknowledgement through ("yes go ahead"), which
# costs one RRF graph query and no LLM call, and adds at most the join grace to
# time-to-first-token (``join_refresh``); on the SDK engine it overlaps the
# query build. A missed recall costs the user the bug in SECRT-2378; the
# asymmetry decides.
WARM_CONTEXT_REFRESH_MIN_WORDS = 3

# Strong refs to started refreshes until they finish (``start_refresh``).
_pending_refresh_tasks: set[asyncio.Task] = set()


def _refresh_recipe() -> SearchConfig:
    """The refresh's edge search: the first turn's recipe with its reranker
    swapped to reciprocal rank fusion.

    Same search methods as graphiti's ``EDGE_HYBRID_SEARCH_CROSS_ENCODER``,
    which the first turn uses — BM25 + cosine + BFS graph traversal — so
    recall *breadth* is identical on both. The first turn keeps the
    per-candidate classifier LLM rerank for the ~10–15% precision lift the
    audit measured on the single most-impactful retrieval per session; a
    refresh uses RRF, with no LLM calls, so re-running recall every
    substantive turn stays cheap. It keeps the BFS method, so a fact
    reachable only by graph expansion from an entity named in the message is
    still surfaced on follow-up turns. Neither sets the limit:
    ``recall.search_facts`` replaces the recipe's default ``limit=10`` with
    ``context_max_facts`` on its own copy.
    """
    base_edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config
    if base_edge_config is None:
        # The recipe always carries an edge_config; this satisfies the type
        # checker and fails loudly if graphiti ever ships a broken recipe.
        raise RuntimeError("EDGE_HYBRID_SEARCH_CROSS_ENCODER has no edge_config")
    edge_config = base_edge_config.model_copy(update={"reranker": EdgeReranker.rrf})
    return EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(
        update={"edge_config": edge_config}
    )


REFRESH_RECIPE = _refresh_recipe()


class PendingRefresh(BaseModel):
    """A follow-up refresh running as a task, and when it started."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    task: "asyncio.Task[str | None]"
    started_at: float


def start_refresh(
    user_id: str | None,
    message: str | None,
    *,
    expert_id: str | None = None,
    force: bool = False,
) -> PendingRefresh | None:
    """Start ``refresh_warm_context`` as a task, to be joined with
    ``join_refresh``; ``None`` when it would skip the turn anyway (no user,
    or a message under the substance gate without ``force``).

    The SDK engine starts it before the query build, so the retrieval
    overlaps compaction, attachments and builder context; everywhere else it
    starts at the join.
    """
    if not user_id or not (force or should_refresh_warm_context(message)):
        return None
    loop = asyncio.get_running_loop()
    task = loop.create_task(
        refresh_warm_context(user_id, message, expert_id=expert_id, force=force),
        name=f"warm-ctx-refresh-{user_id[:12]}",
    )
    # The loop holds tasks weakly, and the caller may be a generator that is
    # closed before it joins: keep a strong reference until the task is done.
    _pending_refresh_tasks.add(task)
    task.add_done_callback(_pending_refresh_tasks.discard)
    return PendingRefresh(task=task, started_at=loop.time())


async def join_refresh(pending: PendingRefresh) -> str | None:
    """The refresh's block if it is done within the join grace, else ``None``.

    ``warm_context_refresh_join_grace_ms`` is the most a follow-up refresh
    may add to time-to-first-token: once the query is ready the turn waits at
    most that long. A refresh still running then is cancelled, logged at
    INFO ("refresh late, skipped") with how long it had run, and the turn
    goes on without a block. A refresh started before the query build had
    the build's time too; one started at the join has the grace alone.
    """
    grace_ms = graphiti_config.warm_context_refresh_join_grace_ms
    try:
        done, _ = await asyncio.wait({pending.task}, timeout=grace_ms / 1000)
    except asyncio.CancelledError:
        pending.task.cancel()
        raise
    if not done:
        pending.task.cancel()
        ran_ms = (asyncio.get_running_loop().time() - pending.started_at) * 1000
        logger.info(
            f"Warm context refresh late, skipped: {ran_ms:.0f} ms since it "
            f"started, join grace {grace_ms} ms"
        )
        return None
    if pending.task.cancelled():
        return None
    return pending.task.result()


async def refresh_warm_context(
    user_id: str | None,
    message: str | None,
    *,
    expert_id: str | None = None,
    force: bool = False,
) -> str | None:
    """Re-fetch warm context on a FOLLOW-UP turn, keyed on the current message.

    The first turn pre-loads memory via ``context.fetch_warm_context``
    (cross-encoder, high precision).  Later turns — a new task mid-session, or
    the turn right after a context compaction — otherwise get no
    deterministic recall and depend on the model choosing to call the memory
    tool, which it often skips (SECRT-2378).  This refresh closes that gap.

    Cost is bounded three ways: ``should_refresh_warm_context`` skips trivial
    turns (unless ``force`` — e.g. just after a compaction, where the current
    message may be short); the fetch runs the RRF recipe (``REFRESH_RECIPE``),
    graph search and one embedding with no per-candidate cross-encoder LLM
    prompts; and it runs for at most ``context_refresh_timeout``. The engines
    run it through ``start_refresh`` / ``join_refresh``, which bound what it
    adds to time-to-first-token by the join grace.

    Returns the ``<temporal_context>`` block, or ``None`` when skipped, empty,
    failed or timed out.
    """
    if not user_id:
        return None
    if not force and not should_refresh_warm_context(message):
        return None
    return await fetch_warm_context(
        user_id,
        message or "",
        expert_id,
        recipe=REFRESH_RECIPE,
        # No ratification hits. The first turn's results passed a
        # per-candidate classifier, so promoting a tentative edge on a hit is
        # earned there; a refresh has no classifier and defaults
        # ``reranker_min_score=0``, so a weak BM25/cosine match could
        # auto-promote an unratified memory, once per substantive turn rather
        # than once per session. Refreshes still retrieve; they don't ratify.
        ratify=False,
        timeout=graphiti_config.context_refresh_timeout,
    )


def should_refresh_warm_context(message: str | None) -> bool:
    """Whether a follow-up user message carries enough signal to re-fetch.

    Pure, deterministic cost gate: only messages with at least
    ``WARM_CONTEXT_REFRESH_MIN_WORDS`` signal units (whitespace words plus
    individually-counted CJK characters) re-run retrieval, keeping the added
    per-turn graph-search cost off trivial acknowledgement turns while still
    firing for whitespace-less languages.
    """
    if not message:
        return False
    return _has_signal_units(message, WARM_CONTEXT_REFRESH_MIN_WORDS)


def _has_signal_units(message: str, threshold: int) -> bool:
    """Whether *message* carries at least *threshold* retrieval signal units.

    Short-circuits: the gate only needs to know whether the threshold is
    reached, and this runs on the pre-query path where a user can paste a
    large log. Counting the whole message would make that a full character
    walk for an answer settled in the first few words.

    A plain ``str.split()`` word count under-counts languages that don't
    separate words with whitespace (Japanese, Chinese, Thai) — a long CJK
    message would score 1 "word" and never pass the gate, silently disabling
    the refresh for those users.  So each CJK/ideographic character counts as
    its own unit and is added to the whitespace-word count of the rest.
    """
    units = 0
    in_word = False
    for ch in message:
        # CJK/ideographic characters count individually; everything else is
        # counted by whitespace-delimited run, so neither is double counted.
        if _is_unspaced_script(ch):
            units += 1
            in_word = False
        elif ch.isspace():
            in_word = False
        elif not in_word:
            in_word = True
            units += 1
        if units >= threshold:
            return True
    return False


def _is_unspaced_script(ch: str) -> bool:
    # CJK Unified Ideographs, Hiragana, Katakana, Hangul, Thai — scripts whose
    # tokens are not whitespace-delimited.  Range checks only, no deps.
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Ext A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0x0E00 <= code <= 0x0E7F  # Thai
    )
