"""Extended-thinking wire support for the baseline (OpenRouter) path.

OpenRouter routes that support extended thinking (Anthropic Claude and
Moonshot Kimi today) expose reasoning through non-OpenAI extension fields
that the OpenAI Python SDK doesn't model:

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
  OpenAI client call.  Returns ``None`` for routes without reasoning
  support (see :func:`_is_reasoning_route`).
"""

from __future__ import annotations

import logging
import time
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

# Coalescing thresholds for ``StreamReasoningDelta`` emission.  OpenRouter's
# Kimi K2.6 endpoint tokenises reasoning at a much finer grain than Anthropic
# (~4,700 deltas per turn in one observed session, vs ~28 for Sonnet); without
# coalescing, every chunk is one Redis ``xadd`` + one SSE frame + one React
# re-render of the non-virtualised chat list, which paint-storms the browser
# main thread and freezes the UI.  Batching into ~32-char / ~40 ms windows
# cuts the event rate ~100x while staying snappy enough that the Reasoning
# collapse still feels live (well under the ~100 ms perceptual threshold).
# Per-delta persistence to ``session.messages`` stays granular — we only
# coalesce the *wire* emission.
_COALESCE_MIN_CHARS = 32
_COALESCE_MAX_INTERVAL_MS = 40.0


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


def _is_reasoning_route(model: str) -> bool:
    """Return True when the route supports OpenRouter's ``reasoning`` extension.

    OpenRouter exposes reasoning tokens via a unified ``reasoning`` request
    param that works on any provider that supports extended thinking —
    currently Anthropic (Claude Opus / Sonnet) and Moonshot (Kimi K2.6 +
    kimi-k2-thinking) advertise it in their ``supported_parameters``.
    Other providers silently drop the field, but we skip it anyway to keep
    the payload tight and avoid confusing cache diagnostics.

    Kept separate from :func:`backend.copilot.baseline.service._is_anthropic_model`
    because ``cache_control`` is strictly Anthropic-specific (Moonshot does
    its own auto-caching), so the two gates must not conflate.

    Both the Claude and Kimi matches are anchored to the provider
    prefix (or to a bare model id with no prefix at all) to avoid
    substring false positives — a custom ``some-other-provider/claude-mock``
    or ``provider/hakimi-large`` configured via
    ``CHAT_FAST_STANDARD_MODEL`` must NOT inherit the reasoning
    extra_body and take a 400 from its upstream.  Recognised shapes:

    * Claude — ``anthropic/`` or ``anthropic.`` provider prefix, or a
      bare ``claude-`` model id with no provider prefix
      (``claude-opus-4.7``, ``anthropic/claude-sonnet-4-6``,
      ``anthropic.claude-3-5-sonnet``).  A non-Anthropic prefix like
      ``someprovider/claude-mock`` is rejected on purpose.
    * Kimi — ``moonshotai/`` provider prefix, or a ``kimi-`` model id
      with no provider prefix (``kimi-k2.6``,
      ``moonshotai/kimi-k2-thinking``).  Like Claude, a non-Moonshot
      prefix is rejected — exception: ``openrouter/kimi-k2.6`` stays
      recognised because ``openrouter/`` is how we route to Moonshot
      today and changing that would be a behaviour regression for
      existing deployments.
    """
    lowered = model.lower()
    if lowered.startswith(("anthropic/", "anthropic.")):
        return True
    if lowered.startswith("moonshotai/"):
        return True
    # ``openrouter/`` historically routes to whatever the default
    # upstream for the model is — for kimi that's Moonshot, so accept
    # ``openrouter/kimi-...`` here.  Other ``openrouter/`` models
    # (e.g. ``openrouter/auto``) fall through to the no-prefix check
    # below and are rejected unless they start with ``claude-`` /
    # ``kimi-`` after the slash, which no real OpenRouter route does.
    if lowered.startswith("openrouter/kimi-"):
        return True
    if "/" in lowered:
        # Any other provider prefix is a custom / non-Anthropic /
        # non-Moonshot route and must not opt into reasoning.  This
        # blocks substring false positives like
        # ``some-provider/claude-mock-v1`` or ``other/kimi-pro``.
        return False
    # No provider prefix — accept bare ``claude-*`` and ``kimi-*`` ids
    # so direct CLI configs (``claude-3-5-sonnet-20241022``,
    # ``kimi-k2-instruct``) keep working.
    return lowered.startswith("claude-") or lowered.startswith("kimi-")


def reasoning_extra_body(model: str, max_thinking_tokens: int) -> dict[str, Any] | None:
    """Build the ``extra_body["reasoning"]`` fragment for the OpenAI client.

    Returns ``None`` for non-reasoning routes and for
    ``max_thinking_tokens <= 0`` (operator kill switch).
    """
    if not _is_reasoning_route(model) or max_thinking_tokens <= 0:
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
        *,
        coalesce_min_chars: int = _COALESCE_MIN_CHARS,
        coalesce_max_interval_ms: float = _COALESCE_MAX_INTERVAL_MS,
    ) -> None:
        self._block_id: str = str(uuid.uuid4())
        self._open: bool = False
        self._session_messages = session_messages
        self._current_row: ChatMessage | None = None
        # Coalescing state — ``_pending_delta`` accumulates reasoning text
        # between wire flushes.  Providers like Kimi K2.6 emit very fine-
        # grained chunks; batching them reduces Redis ``xadd`` + SSE + React
        # re-render load by ~100x for equivalent text output.  Tuning knobs
        # are kwargs so tests can disable coalescing (``=0``) for
        # deterministic event assertions.
        self._coalesce_min_chars = coalesce_min_chars
        self._coalesce_max_interval_ms = coalesce_max_interval_ms
        self._pending_delta: str = ""
        self._last_flush_monotonic: float = 0.0

    @property
    def is_open(self) -> bool:
        return self._open

    def on_delta(self, delta: ChoiceDelta) -> list[StreamBaseResponse]:
        """Return events for the reasoning text carried by *delta*.

        Empty list when the chunk carries no reasoning payload, so this is
        safe to call on every chunk without guarding at the call site.

        Persistence (when a session message list is attached) stays
        per-delta so the DB row's content always equals the concatenation
        of wire deltas at every chunk boundary, independent of the
        coalescing window.  Only the wire emission is batched.
        """
        ext = OpenRouterDeltaExtension.from_delta(delta)
        text = ext.visible_text()
        if not text:
            return []
        events: list[StreamBaseResponse] = []
        # First reasoning text in this block — emit Start + the first Delta
        # atomically so the frontend Reasoning collapse renders immediately
        # rather than waiting for the coalesce window to elapse.  Subsequent
        # chunks buffer into ``_pending_delta`` and only flush when the
        # char/time thresholds trip.
        # Sample the monotonic clock exactly once per chunk — at ~4,700
        # chunks per turn, folding the two calls into one cuts ~4,700
        # syscalls off the hot path without changing semantics.
        now = time.monotonic()
        if not self._open:
            events.append(StreamReasoningStart(id=self._block_id))
            events.append(StreamReasoningDelta(id=self._block_id, delta=text))
            self._open = True
            self._last_flush_monotonic = now
            if self._session_messages is not None:
                self._current_row = ChatMessage(role="reasoning", content=text)
                self._session_messages.append(self._current_row)
            return events

        # Persist per-delta (no coalescing here — the session snapshot stays
        # consistent at every chunk boundary, independent of the wire
        # coalesce window).
        if self._current_row is not None:
            self._current_row.content = (self._current_row.content or "") + text

        self._pending_delta += text
        if self._should_flush_pending(now):
            events.append(
                StreamReasoningDelta(id=self._block_id, delta=self._pending_delta)
            )
            self._pending_delta = ""
            self._last_flush_monotonic = now
        return events

    def _should_flush_pending(self, now: float) -> bool:
        """Return True when the accumulated delta should be emitted now.

        *now* is the monotonic timestamp sampled by the caller so the
        clock is read at most once per chunk (the flush-timestamp update
        reuses the same value).
        """
        if not self._pending_delta:
            return False
        if len(self._pending_delta) >= self._coalesce_min_chars:
            return True
        elapsed_ms = (now - self._last_flush_monotonic) * 1000.0
        return elapsed_ms >= self._coalesce_max_interval_ms

    def close(self) -> list[StreamBaseResponse]:
        """Emit ``StreamReasoningEnd`` for the open block (if any) and rotate.

        Idempotent — returns ``[]`` when no block is open.  Drains any
        still-buffered delta first so the frontend never loses tail text
        from the coalesce window.  The id rotation guarantees the next
        reasoning block starts with a fresh id rather than reusing one
        already closed on the wire.  The persisted row is not removed —
        it stays in ``session_messages`` as the durable record of what
        was reasoned.
        """
        if not self._open:
            return []
        events: list[StreamBaseResponse] = []
        if self._pending_delta:
            events.append(
                StreamReasoningDelta(id=self._block_id, delta=self._pending_delta)
            )
            self._pending_delta = ""
        events.append(StreamReasoningEnd(id=self._block_id))
        self._open = False
        self._block_id = str(uuid.uuid4())
        self._current_row = None
        return events
