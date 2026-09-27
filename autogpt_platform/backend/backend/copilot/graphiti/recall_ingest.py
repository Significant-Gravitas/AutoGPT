"""How ingestion keeps graphiti from writing over a forget, and which
earlier episodes it shows graphiti's extraction (``previous_episode_uuids``).

Two things could write over a forget. An ingestion that read a fact before
a forget landed saves its older copy after it (graphiti saves what it read:
``SET r = edge``); the graph's write lock rules that out while Redis answers
and the holder keeps its lease, since a forget and an ingestion then never
overlap (``scope_lock.py``; ``graphiti/AGENTS.md`` has the windows with no
bound). And a later ingestion resolves each new statement against the edges
between the same entities, forgotten ones included: if its model calls the
statement a duplicate of a forgotten edge, graphiti appends the episode to
that edge and, because no edge type is named ``[forgotten]``, clears every
attribute on it, the forget's marker and audit copies among them
(``edge_operations.resolve_extracted_edge``, then its bulk save's
``SET r = edge``); if it calls it a contradiction, it stamps the forgotten
edge's ``invalid_at``.

So every graphiti client is built with ``ForgetAwareLLMClient``
(``client._build_graphiti``): whatever graphiti's model answers, an edge
whose text is ``recall.FORGOTTEN_FACT`` is neither a duplicate nor a
contradiction. graphiti then saves a restated fact as a new live edge in the
same ``add_episode``, as it would any new fact, and its model's dedup never
touches the forgotten one: no second extraction, and nothing to repair
afterwards. One path runs before the model and so past this guard:
graphiti's exact-text match reuses an edge whose text equals the new
statement's, lower-cased and whitespace-collapsed, so a statement that reads
``[forgotten]`` appends its episode's uuid to the forgotten edge's
``episodes``. The edge keeps its marker and audit copies and stays out of
recall, and the episode, citing it, is hidden.

The guard reads graphiti's prompt, so it fails closed. A candidate printed
with the placeholder as its fact or name is forgotten wherever it appears,
inside the two candidate lists or not, so renaming a tag cannot hide one.
And when the prompt shows the placeholder anywhere, both lists must be
there, once each, and parse, and every placeholder must belong to a
candidate the guard can read; otherwise the answer names no edge at all and
the statement is saved as new (a duplicate at worst, never a merge into a
forgotten edge). Every prompt it cannot read in full is logged at error
once per process for each shape and reason, so a graphiti upgrade that
changes the prompt is noticed even before a forgotten edge shows up in it.
"""

import ast
import logging
import re
from datetime import datetime
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.tracer import Tracer
from pydantic import BaseModel, TypeAdapter, ValidationError

from .recall import FORGOTTEN_FACT, recallable_episodes

logger = logging.getLogger(__name__)

# The two candidate lists graphiti's edge dedup prompt shows
# (``prompts/dedupe_edges.resolve_edge``), each a Python list of
# ``{'idx': n, 'fact': ...}`` with one idx numbering across both.
_LISTS = ("EXISTING FACTS", "FACT INVALIDATION CANDIDATES")
_LIST = re.compile(
    r"<(EXISTING FACTS|FACT INVALIDATION CANDIDATES)>\s*(.*?)\s*</\1>", re.DOTALL
)
# One candidate as graphiti prints it, wherever it is: a flat dict literal.
_ENTRY = re.compile(r"\{[^{}]*\}")
# A prompt's tags, in order: its shape, for the once-per-shape error log.
_TAG = re.compile(r"</?([A-Z][A-Z _]*)>")
_REPORTED_SHAPES: set[tuple[str, ...]] = set()
# What ``ast.literal_eval`` and the validation raise on text they cannot read.
_UNREADABLE = (ValueError, TypeError, SyntaxError, MemoryError, RecursionError)


class _Candidate(BaseModel):
    """One candidate edge as graphiti's dedup prompt prints it."""

    idx: int
    fact: str | None = None
    name: str | None = None

    @property
    def forgotten(self) -> bool:
        return FORGOTTEN_FACT in (self.fact, self.name)


_CANDIDATE_LIST = TypeAdapter(list[_Candidate])


class ForgetAwareLLMClient(LLMClient):
    """``inner``, except that a forgotten edge is never a duplicate or a
    contradiction in graphiti's edge dedup answers."""

    def __init__(self, inner: LLMClient) -> None:
        super().__init__(inner.config)
        self.inner = inner
        self.token_tracker = inner.token_tracker

    def set_tracer(self, tracer: Tracer) -> None:
        super().set_tracer(tracer)
        self.inner.set_tracer(tracer)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, Any]:
        forgotten = (
            forgotten_candidates(messages) if response_model is EdgeDuplicate else set()
        )
        answer = await self.inner.generate_response(
            messages,
            response_model,
            max_tokens,
            model_size,
            group_id,
            prompt_name,
            attribute_extraction=attribute_extraction,
        )
        return answer if forgotten == set() else off_forgotten(answer, forgotten)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        """Unused: ``generate_response`` hands every call to ``inner``."""
        return await self.inner._generate_response(
            messages, response_model, max_tokens, model_size
        )


async def previous_episode_uuids(
    driver: GraphDriver,
    group_id: str,
    reference_time: datetime,
    source: EpisodeType,
) -> list[str]:
    """The earlier episodes ``add_episode`` may show its extraction prompts.

    graphiti's own pick (``retrieve_episodes``: the ``RELEVANT_SCHEMA_LIMIT``
    newest of the same source up to ``reference_time``) cannot see a forget,
    so ingestion passes this one: the same pick of recallable episodes,
    oldest first. Never raises: on a failed read extraction gets no earlier
    episodes, not graphiti's unfiltered pick, and the write still happens.

    Inherited limitation: on FalkorDB 4.x the indexed ``valid_at <=
    $reference_time`` range can admit an episode dated just after the
    cut-off (a direct comparison of the same values says it should not),
    so, exactly like graphiti's own ``retrieve_episodes``, the newest
    episode here may postdate ``reference_time``. Not fixed here.
    """
    try:
        records = await recallable_episodes(
            driver, group_id, reference_time, RELEVANT_SCHEMA_LIMIT, source.value
        )
    except Exception:
        logger.warning(
            f"Prior-episode read failed for group {group_id[:12]}; "
            "extracting without earlier episodes",
            exc_info=True,
        )
        return []
    return [str(record["uuid"]) for record in reversed(records)]


def forgotten_candidates(messages: list[Message]) -> set[int] | None:
    """The idx of every forgotten edge an edge dedup prompt shows; None, so
    that the answer names no edge, when the prompt shows the placeholder but
    cannot be read in full."""
    text = "\n".join(message.content for message in messages)
    listed = _listed(text)
    if listed is None:
        _report_unreadable(text, "its candidate lists cannot be read")
    if FORGOTTEN_FACT not in text:
        return set()
    shown = _shown(text)
    if shown is None:
        _report_unreadable(text, "it shows the placeholder outside a candidate")
    if listed is None or shown is None:
        return None
    return {c.idx for c in listed if c.forgotten} | shown


def _listed(text: str) -> list[_Candidate] | None:
    """Every candidate in the two lists; None unless each list is there
    once and reads as candidates with a fact each."""
    lists = _LIST.findall(text)
    if sorted(tag for tag, _ in lists) != sorted(_LISTS):
        return None
    try:
        listed = [
            candidate
            for _, body in lists
            for candidate in _CANDIDATE_LIST.validate_python(ast.literal_eval(body))
        ]
    except _UNREADABLE:
        return None
    return None if any(c.fact is None for c in listed) else listed


def _shown(text: str) -> set[int] | None:
    """The idx of every candidate anywhere in ``text``, in a list or not,
    whose fact or name is the placeholder; None when the placeholder also
    appears anywhere else."""
    entries = [entry for entry in _ENTRY.findall(text) if FORGOTTEN_FACT in entry]
    placed = sum(entry.count(FORGOTTEN_FACT) for entry in entries)
    if placed != text.count(FORGOTTEN_FACT):
        return None
    try:
        shown = [_Candidate.model_validate(ast.literal_eval(e)) for e in entries]
    except _UNREADABLE:
        return None
    return {c.idx for c in shown} if all(c.forgotten for c in shown) else None


def _report_unreadable(text: str, why: str) -> None:
    """Log, at error and once per process for each reason and shape (the
    prompt's tags), a dedup prompt the guard cannot read in full."""
    tags = _TAG.findall(text)
    shape = (why, *tags)
    if shape in _REPORTED_SHAPES:
        return
    _REPORTED_SHAPES.add(shape)
    logger.error(
        "graphiti's edge dedup prompt no longer reads as recall_ingest expects: "
        f"{why} (tags {tags}); while it does not, an answer that may name a "
        "forgotten edge names none, so restated facts are saved as new"
    )


def off_forgotten(answer: dict[str, Any], forgotten: set[int] | None) -> dict[str, Any]:
    """``answer`` naming no forgotten edge; naming no edge at all when which
    ones are forgotten could not be read, so the fact is saved as new."""
    try:
        decided = EdgeDuplicate.model_validate(answer)
    except ValidationError:
        return answer
    if forgotten is None:
        logger.warning("Edge dedup prompt unreadable; the new fact is kept as new")
    kept = EdgeDuplicate(
        duplicate_facts=_keep(decided.duplicate_facts, forgotten),
        contradicted_facts=_keep(decided.contradicted_facts, forgotten),
    )
    if kept != decided:
        logger.info("graphiti's model named a forgotten fact in edge dedup; ignored")
    return {**answer, **kept.model_dump()}


def _keep(idxs: list[int], forgotten: set[int] | None) -> list[int]:
    return [i for i in idxs if forgotten is not None and i not in forgotten]
