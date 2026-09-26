"""How ingestion keeps graphiti from writing over a forget.

Two things could. An ingestion that read a fact before a forget landed saves
its older copy after it (graphiti saves what it read: ``SET r = edge``); the
graph's write lock rules that out, since a forget and an ingestion never
overlap (``scope_lock.py``). And a later ingestion resolves each new
statement against the edges between the same entities, forgotten ones
included: if its model calls the statement a duplicate of a forgotten edge,
graphiti appends the episode to that edge and, because no edge type is named
``[forgotten]``, clears every attribute on it, the forget's marker and audit
copies among them (``edge_operations.resolve_extracted_edge``, then its bulk
save's ``SET r = edge``); if it calls it a contradiction, it stamps the
forgotten edge's ``invalid_at``.

So every graphiti client is built with ``ForgetAwareLLMClient``
(``client._build_graphiti``): whatever graphiti's model answers, an edge
whose text is ``recall.FORGOTTEN_FACT`` is neither a duplicate nor a
contradiction. graphiti then saves a restated fact as a new live edge in the
same ``add_episode``, as it would any new fact, and never touches the
forgotten one: no second extraction, and nothing to repair afterwards.
"""

import ast
import logging
import re
from typing import Any

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.models import Message
from graphiti_core.tracer import Tracer
from pydantic import BaseModel, TypeAdapter, ValidationError

from .recall import FORGOTTEN_FACT

logger = logging.getLogger(__name__)

# The two candidate lists graphiti's edge dedup prompt shows
# (``prompts/dedupe_edges.resolve_edge``), each a Python list of
# ``{'idx': n, 'fact': ...}`` with one idx numbering across both.
_CANDIDATES = re.compile(
    r"<(EXISTING FACTS|FACT INVALIDATION CANDIDATES)>\s*(.*?)\s*</\1>", re.DOTALL
)


class _Candidate(BaseModel):
    idx: int
    fact: str


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


def forgotten_candidates(messages: list[Message]) -> set[int] | None:
    """The idx of every forgotten edge an edge dedup prompt shows; None when
    the prompt shows one but its candidate lists cannot be read."""
    text = "\n".join(message.content for message in messages)
    if FORGOTTEN_FACT not in text:
        return set()
    sections = _CANDIDATES.findall(text)
    if not sections:
        return None
    try:
        candidates = [
            candidate
            for _, body in sections
            for candidate in _CANDIDATE_LIST.validate_python(ast.literal_eval(body))
        ]
    except (ValueError, TypeError, SyntaxError):
        return None
    return {c.idx for c in candidates if c.fact == FORGOTTEN_FACT}


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
