"""Select complete proof and nearby context within the chat review budget."""

import json

from backend.copilot.hidden_rows import visible_to_model
from backend.copilot.model import ChatMessage
from backend.data.skill_learning import MAX_OUTCOME_SIGNALS

from .contract import (
    VERIFYING_SIGNAL_KINDS,
    EvidenceBundle,
    EvidenceSpan,
    SourceRevision,
)
from .signals import message_ref

MAX_CONTEXT_SPAN_CHARS = 2_000


def build_chat_evidence(
    source: SourceRevision,
    messages: list[ChatMessage],
    *,
    last_sequence: int,
    max_chars: int,
) -> EvidenceBundle:
    # The platform's notes to the owner, such as why a scheduled run was
    # skipped, go to no model: learnt from, "usage limit reached" would read
    # as something about the user.
    ordered = sorted(
        (
            m
            for m in visible_to_model(messages)
            if m.sequence is not None and 0 <= m.sequence <= last_sequence
        ),
        key=lambda m: m.sequence or 0,
    )
    positions = {message_ref(m): i for i, m in enumerate(ordered)}
    spans, clipped = _bounded_spans(source, ordered, last_sequence, max_chars)
    present = {span.ref for span in spans}
    omitted = sorted(
        {r.ref for r in source.evidence_refs if r.ref not in present} | set(clipped)
    )
    return EvidenceBundle(
        source=source,
        spans=sorted(spans, key=lambda span: positions[span.ref]),
        omitted_refs=omitted,
        clipped_refs=clipped,
    )


def verification_sequences(source: SourceRevision, last_sequence: int) -> set[int]:
    return {
        int(s.ref[4:])
        for s in source.outcome_signals[:MAX_OUTCOME_SIGNALS]
        if s.kind in VERIFYING_SIGNAL_KINDS
        and s.ref.startswith("msg:")
        and s.ref[4:].isascii()
        and s.ref[4:].isdigit()
        and int(s.ref[4:]) <= last_sequence
    }


def verification_context(sequences: set[int]) -> set[int]:
    return {n - distance for n in sequences for distance in range(3) if n >= distance}


def _bounded_spans(
    source: SourceRevision,
    messages: list[ChatMessage],
    last_sequence: int,
    max_chars: int,
) -> tuple[list[EvidenceSpan], list[str]]:
    outcomes = {s.ref: s.kind for s in source.outcome_signals}
    verifying = {
        s.ref for s in source.outcome_signals if s.kind in VERIFYING_SIGNAL_KINDS
    }
    context = verification_context(verification_sequences(source, last_sequence))
    prioritized = sorted(
        reversed(messages),
        key=lambda m: (message_ref(m) not in verifying, m.sequence not in context),
    )
    spans: list[EvidenceSpan] = []
    full_texts: dict[str, str] = {}
    clipped: list[str] = []
    used = 0
    for message in prioritized:
        text = _message_text(message)
        if not text:
            continue
        ref = message_ref(message)
        body = text[: max_chars if ref in verifying else MAX_CONTEXT_SPAN_CHARS]
        if len(body) < len(text):
            clipped.append(ref)
        if used + len(body) > max_chars:
            clipped.append(ref)
            continue
        used += len(body)
        full_texts[ref] = text
        spans.append(
            EvidenceSpan(
                ref=ref, role=message.role, text=body, outcome=outcomes.get(ref)
            )
        )
    return _restore_clipped(spans, full_texts, clipped, max_chars - used)


def _restore_clipped(
    spans: list[EvidenceSpan],
    full_texts: dict[str, str],
    clipped: list[str],
    spare: int,
) -> tuple[list[EvidenceSpan], list[str]]:
    """Spend the budget the per-span cap left unused on the spans it clipped.

    The user's messages come first, since they state the task and its rules,
    then the rest in their selection order. A real coding turn otherwise
    reaches review with its request and code cut to the cap even when most
    of the budget is unused, and the reviewer can't confirm what passed.
    """
    restored = {}
    for span in sorted(spans, key=lambda s: s.role != "user"):
        missing = len(full_texts[span.ref]) - len(span.text)
        if spare <= 0 or missing <= 0:
            continue
        added = min(missing, spare)
        spare -= added
        restored[span.ref] = full_texts[span.ref][: len(span.text) + added]
    whole = {ref for ref, text in restored.items() if text == full_texts[ref]}
    return (
        [
            (
                span.model_copy(update={"text": restored[span.ref]})
                if span.ref in restored
                else span
            )
            for span in spans
        ],
        [ref for ref in clipped if ref not in whole],
    )


def _message_text(message: ChatMessage) -> str:
    if message.role not in ("user", "assistant", "tool"):
        return ""
    text = (message.content or "").strip()
    if message.role == "assistant" and message.tool_calls:
        text = "\n".join(
            part for part in (text, json.dumps(message.tool_calls)) if part
        )
    return text
