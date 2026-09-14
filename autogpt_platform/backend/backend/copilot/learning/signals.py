"""Deterministic evidence and outcome signals from one conversation turn.

The capture step never asks a model to write a lesson. It records bounded
references to what actually happened and only a narrow set of unambiguous
signals:

* a tool result whose *typed* payload reports a checkable outcome (an
  exit code, a completed execution, a passing validation) — merely
  returning without an error is not one;
* the user's own explicit, unqualified outcome report or acceptance
  (negated, questioning, conditional, or quoted wording never counts);
* the user's explicit request to learn the procedure.

Anything else is context for the reviewer, not evidence of success. No
chat wording ever creates an approval checkpoint; approval-aware sources
receive that only from their adapter's event.
"""

from __future__ import annotations

import json
import re

from backend.copilot.model import ChatMessage

from .contract import EvidenceRef, OutcomeSignal, SignalKind

_CONFIRMATION_RE = re.compile(
    r"\b(that worked|it worked|works now|that fixed it|this fixed it|"
    r"confirmed working|the import (?:worked|succeeded)|"
    r"i (?:checked|verified) (?:it|this|the (?:output|result|import)) and it (?:works|is correct))\b",
    re.IGNORECASE,
)
_ACCEPTANCE_RE = re.compile(
    r"\b(i approve (?:this|that|it)|approved,? (?:thanks|go ahead)|"
    r"i accept (?:this|that|it)|accepted,? (?:thanks|go ahead)|ship it)\b",
    re.IGNORECASE,
)
_LEARN_REQUEST_RE = re.compile(
    r"\b(learn (?:this|that|from this)|save (?:this|that) as a skill|"
    r"remember how to do this|make (?:this|that) a skill|"
    r"turn (?:this|that) into a skill)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|never|no|none|nothing|hasn'?t|haven'?t|didn'?t|doesn'?t|don'?t|"
    r"isn'?t|wasn'?t|aren'?t|weren'?t|can'?t|couldn'?t|won'?t|wouldn'?t|"
    r"unless|until|before|neither|nor|hardly)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r"\b(if|whether|once|when|assuming|suppose|in case|should)\b", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+|\n+")
_QUOTED_RE = re.compile(r"(\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`|“[^”\n]*”)")


def message_ref(message: ChatMessage) -> str:
    seq = message.sequence if message.sequence is not None else "?"
    return f"msg:{seq}"


def _explicit_match(pattern: re.Pattern[str], text: str) -> bool:
    """True only when the pattern matches an unambiguous sentence.

    A sentence is rejected when it is a question, when a negation or a
    conditional word precedes the match inside the sentence, or when the
    match sits inside a quotation (an example, not a report).
    """
    for raw in _SENTENCE_SPLIT_RE.split(text):
        sentence = raw.strip()
        if not sentence or "?" in sentence:
            continue
        unquoted = _QUOTED_RE.sub(" ", sentence)
        match = pattern.search(unquoted)
        if match is None:
            continue
        prefix = unquoted[: match.start()]
        if _NEGATION_RE.search(prefix) or _CONDITIONAL_RE.search(prefix):
            continue
        return True
    return False


# Tool response ``type`` values whose payload carries a checkable outcome.
# The checker returns ``True`` when that outcome reports success, ``False``
# when it reports failure, ``None`` when the payload is not an outcome.
def _tool_outcome(payload: dict[str, object]) -> bool | None:
    kind = str(payload.get("type") or "")
    if kind == "error" or payload.get("success") is False:
        return False
    if kind == "block_output":
        return payload.get("success") is True
    if kind == "bash_exec":
        return payload.get("exit_code") == 0 and payload.get("timed_out") is not True
    if kind == "mcp_tool_output":
        return payload.get("success") is True
    if kind == "agent_builder_validation_result":
        return payload.get("valid") is True
    if kind == "agent_builder_fix_result":
        return payload.get("valid_after_fix") is True
    if kind == "agent_builder_saved":
        return True
    if kind == "agent_output":
        execution = payload.get("execution")
        if isinstance(execution, dict):
            status = str(execution.get("status") or "").upper()
            if status == "COMPLETED":
                return True
            if status in ("FAILED", "TERMINATED"):
                return False
        return None
    return None


def _parse_tool_payload(content: str | None) -> dict[str, object] | None:
    if not content:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _tool_signal(
    message: ChatMessage, ref: str
) -> tuple[EvidenceRef, OutcomeSignal | None]:
    payload = _parse_tool_payload(message.content)
    kind_label = str(payload.get("type") or "tool") if payload else "tool"
    evidence = EvidenceRef(kind="tool", ref=ref, label=f"tool result ({kind_label})")
    if payload is None:
        return evidence, None
    outcome = _tool_outcome(payload)
    if outcome is None:
        return evidence, None
    signal_kind: SignalKind = "tool_result" if outcome else "tool_error"
    label = f"checked outcome: {kind_label} {'succeeded' if outcome else 'failed'}"
    return evidence, OutcomeSignal(kind=signal_kind, ref=ref, label=label)


def extract_turn_signals(
    messages: list[ChatMessage],
) -> tuple[list[EvidenceRef], list[OutcomeSignal]]:
    """Bounded evidence references and outcome signals for one turn."""
    refs: list[EvidenceRef] = []
    signals: list[OutcomeSignal] = []
    for message in messages:
        ref = message_ref(message)
        if message.role == "tool":
            evidence, signal = _tool_signal(message, ref)
            refs.append(evidence)
            if signal is not None:
                signals.append(signal)
            continue
        if message.role == "assistant" and (message.content or "").strip():
            refs.append(EvidenceRef(kind="assistant", ref=ref, label="assistant reply"))
            continue
        if message.role != "user":
            continue
        text = message.content or ""
        refs.append(EvidenceRef(kind="user", ref=ref, label="user message"))
        signals.extend(user_signals(text, ref))
    return refs, signals


def user_signals(text: str, ref: str) -> list[OutcomeSignal]:
    """Explicit, unqualified user reports only (see module docstring)."""
    signals: list[OutcomeSignal] = []
    if _explicit_match(_CONFIRMATION_RE, text):
        signals.append(
            OutcomeSignal(
                kind="user_confirmation", ref=ref, label="User confirmed it worked"
            )
        )
    if _explicit_match(_ACCEPTANCE_RE, text):
        signals.append(
            OutcomeSignal(
                kind="accepted_artifact", ref=ref, label="Result accepted once"
            )
        )
    if _explicit_match(_LEARN_REQUEST_RE, text):
        signals.append(
            OutcomeSignal(kind="learn_request", ref=ref, label="Requested by user")
        )
    return signals


def is_learn_request(text: str) -> bool:
    return _explicit_match(_LEARN_REQUEST_RE, text or "")
