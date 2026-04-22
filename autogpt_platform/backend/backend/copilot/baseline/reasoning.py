"""Extended-thinking wire support for the baseline (OpenRouter) path.

Anthropic routes on OpenRouter expose extended thinking through
non-OpenAI extension fields that the OpenAI Python SDK doesn't model:

* ``reasoning`` (legacy string) — enabled by ``include_reasoning: true``.
* ``reasoning_content`` — DeepSeek / some OpenRouter routes.
* ``reasoning_details`` — structured list shipped with the unified
  ``reasoning`` request param.

This module keeps the wire-level concerns in one place:

* :class:`OpenRouterDeltaExtension` validates the extension dict pulled off
  ``ChoiceDelta.model_extra`` into typed pydantic models — no ``getattr`` +
  ``isinstance`` duck-typing at the call site.
* :class:`BaselineReasoningEmitter` owns the reasoning block lifecycle for
  one streaming round and emits ``StreamReasoning*`` events so the caller
  only has to plumb the events into its pending queue.
* :func:`reasoning_extra_body` builds the ``extra_body`` fragment for the
  OpenAI client call.  Returns ``None`` on non-Anthropic routes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.copilot.model import ChatMessage
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
)

logger = logging.getLogger(__name__)


_VISIBLE_REASONING_TYPES = frozenset({"reasoning.text", "reasoning.summary"})


class ReasoningDetail(BaseModel):
    """One entry in OpenRouter's ``reasoning_details`` list.

    OpenRouter ships ``type: "reasoning.text"`` / ``"reasoning.summary"`` /
    ``"reasoning.encrypted"`` entries.  Only the first two carry
    user-visible text; encrypted entries are opaque and omitted from the
    rendered collapse.  Unknown future types are tolerated (``extra="ignore"``)
    so an upstream addition doesn't crash the stream — but their ``text`` /
    ``summary`` fields are NOT surfaced because they may carry provider
    metadata rather than user-visible reasoning (see
    :attr:`visible_text`).
    """

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    text: str | None = None
    summary: str | None = None

    @property
    def visible_text(self) -> str:
        """Return the human-readable text for this entry, or ``""``.

        Only entries with a recognised reasoning type (``reasoning.text`` /
        ``reasoning.summary``) surface text; unknown or encrypted types
        return an empty string even if they carry a ``text`` /
        ``summary`` field, to guard against future provider metadata
        being rendered as reasoning in the UI.  Entries missing a
        ``type`` are treated as text (pre-``reasoning_details`` OpenRouter
        payloads omit the field).
        """
        if self.type is not None and self.type not in _VISIBLE_REASONING_TYPES:
            return ""
        return self.text or self.summary or ""


class OpenRouterDeltaExtension(BaseModel):
    """Non-OpenAI fields OpenRouter adds to streaming deltas.

    Instantiate via :meth:`from_delta` which pulls the extension dict off
    ``ChoiceDelta.model_extra`` (where pydantic v2 stashes fields that
    aren't part of the declared schema) and validates it through this
    model.  That keeps the parser honest — malformed entries surface as
    validation errors rather than silent ``None``-coalesce bugs — and
    avoids the ``getattr`` + ``isinstance`` duck-typing the earlier inline
    extractor relied on.
    """

    model_config = ConfigDict(extra="ignore")

    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[ReasoningDetail] = Field(default_factory=list)

    @classmethod
    def from_delta(cls, delta: ChoiceDelta) -> "OpenRouterDeltaExtension":
        """Build an extension view from ``delta.model_extra``.

        Malformed provider payloads (e.g. ``reasoning_details`` shipped as
        a string rather than a list) surface as a ``ValidationError`` which
        is logged and swallowed — returning an empty extension so the rest
        of the stream (valid text / tool calls) keeps flowing.  An optional
        feature's corrupted wire data must never abort the whole stream.
        """
        try:
            return cls.model_validate(delta.model_extra or {})
        except ValidationError as exc:
            logger.warning(
                "[Baseline] Dropping malformed OpenRouter reasoning payload: %s",
                exc,
            )
            return cls()

    def visible_text(self) -> str:
        """Concatenated reasoning text, pulled from whichever channel is set.

        Priority: the legacy ``reasoning`` string, then DeepSeek's
        ``reasoning_content``, then the concatenation of text-bearing
        entries in ``reasoning_details``.  Only one channel is set per
        provider in practice; the priority order just makes the fallback
        deterministic if a provider ever emits multiple.
        """
        if self.reasoning:
            return self.reasoning
        if self.reasoning_content:
            return self.reasoning_content
        return "".join(d.visible_text for d in self.reasoning_details)


def reasoning_extra_body(model: str, max_thinking_tokens: int) -> dict[str, Any] | None:
    """Build the ``extra_body["reasoning"]`` fragment for the OpenAI client.

    Returns ``None`` for non-Anthropic routes (other OpenRouter providers
    ignore the field but we skip it anyway to keep the payload minimal)
    and for ``max_thinking_tokens <= 0`` (operator kill switch).
    """
    # Imported lazily to avoid pulling service.py at module load — service.py
    # imports this module, and the lazy import keeps the dependency one-way.
    from backend.copilot.baseline.service import _is_anthropic_model

    if not _is_anthropic_model(model) or max_thinking_tokens <= 0:
        return None
    return {"reasoning": {"max_tokens": max_thinking_tokens}}


class BaselineReasoningEmitter:
    """Owns the reasoning block lifecycle for one streaming round.

    Two concerns live here, both driven by the same state machine:

    1. **Wire events.**  The AI SDK v6 wire format pairs every
       ``reasoning-start`` with a matching ``reasoning-end`` and treats
       reasoning / text / tool-use as distinct UI parts that must not
       interleave.
    2. **Session persistence.**  ``ChatMessage(role="reasoning")`` rows in
       ``session.messages`` are what
       ``convertChatSessionToUiMessages.ts`` folds into the assistant
       bubble as ``{type: "reasoning"}`` UI parts on reload and on
       ``useHydrateOnStreamEnd`` swaps.  Without them the live-streamed
       reasoning parts get overwritten by the hydrated (reasoning-less)
       message list the moment the stream ends.  Mirrors the SDK path's
       ``acc.reasoning_response`` pattern so both routes render the same
       way on reload.

    Pass ``session_messages`` to enable persistence; omit for pure
    wire-emission (tests, scratch callers).  On first reasoning delta a
    fresh ``ChatMessage(role="reasoning")`` is appended and mutated
    in-place as further deltas arrive; :meth:`close` drops the reference
    but leaves the appended row intact.
    """

    def __init__(
        self,
        session_messages: list[ChatMessage] | None = None,
    ) -> None:
        self._block_id: str = str(uuid.uuid4())
        self._open: bool = False
        self._session_messages = session_messages
        self._current_row: ChatMessage | None = None

    @property
    def is_open(self) -> bool:
        return self._open

    def on_delta(self, delta: ChoiceDelta) -> list[StreamBaseResponse]:
        """Return events for the reasoning text carried by *delta*.

        Empty list when the chunk carries no reasoning payload, so this is
        safe to call on every chunk without guarding at the call site.
        Persistence (when a session message list is attached) happens in
        lockstep with emission so the row's content stays equal to the
        concatenated deltas at every delta boundary.
        """
        ext = OpenRouterDeltaExtension.from_delta(delta)
        text = ext.visible_text()
        if not text:
            return []
        events: list[StreamBaseResponse] = []
        if not self._open:
            events.append(StreamReasoningStart(id=self._block_id))
            self._open = True
            if self._session_messages is not None:
                self._current_row = ChatMessage(role="reasoning", content="")
                self._session_messages.append(self._current_row)
        events.append(StreamReasoningDelta(id=self._block_id, delta=text))
        if self._current_row is not None:
            self._current_row.content = (self._current_row.content or "") + text
        return events

    def close(self) -> list[StreamBaseResponse]:
        """Emit ``StreamReasoningEnd`` for the open block (if any) and rotate.

        Idempotent — returns ``[]`` when no block is open.  The id rotation
        guarantees the next reasoning block starts with a fresh id rather
        than reusing one already closed on the wire.  The persisted row is
        not removed — it stays in ``session_messages`` as the durable
        record of what was reasoned.
        """
        if not self._open:
            return []
        event = StreamReasoningEnd(id=self._block_id)
        self._open = False
        self._block_id = str(uuid.uuid4())
        self._current_row = None
        return [event]
