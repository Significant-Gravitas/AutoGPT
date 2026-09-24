"""Pending-approval rows for gated tool calls, on the platform's HITL rails.

``run_block`` already parks sensitive block executions in ``PendingHumanReview``
under a ``copilot-session-<id>`` key, and the chat renders every such row via
``extractGraphExecId`` -> ``CopilotPendingReviews``. Writing the same shape for
a tool call inherits the Home "Needs You" row, the awaiting-review alert and
the approve/reject endpoint; the payload carries what the approval card reads.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
)
from backend.copilot.model import ChatSession

# Private name on purpose: the alternative is editing sharing/models.py to
# add an alias, which shifts a docstring the secrets baseline trips on.
from backend.copilot.sharing.models import _redact_secret_keys
from backend.data.db_accessors import review_db

from .headline import Headline, headline_for
from .policy import DEFAULT_MODE, effect_for

if TYPE_CHECKING:
    from .subject import Subject as GateSubject

logger = logging.getLogger(__name__)

# Keep the stored payload small: @@agptfile: references are expanded before the
# tool handler runs, so an argument can arrive holding a whole file.
_MAX_ARG_CHARS = 4_000

GATE_NODE_PREFIX = f"{COPILOT_NODE_PREFIX}gate-"

ReasonKind = Literal["mode", "subject", "supervisor", "rule", "spend", "content"]


class Subject(BaseModel):
    kind: str = "tool"
    key: str
    name: str
    effect: str
    irreversible: bool = False
    # A block's fields are labelled from its own input schema.
    block_id: str | None = None


class FieldLabel(BaseModel):
    key: str
    label: str


class GateReviewPayload(BaseModel):
    """What the approval card, Home and the channels render: never the raw call."""

    tool: str
    arguments: dict[str, Any]
    clipped: list[str] = []
    fields: list[FieldLabel] = []
    tool_call_id: str = ""
    turn: int = 0
    mode: str | None = None
    subject: Subject
    reason: str = ""
    reason_kind: ReasonKind = "mode"
    # Only a card naming a subject can set a rule on it.
    chat_rules_allowed: list[Literal["allow", "judge"]] = []
    headline: Headline


# An approval must not run a call long after the user gave it; the answered
# card's turn normally runs it within seconds.
APPROVAL_TTL = timedelta(hours=1)


def session_exec_id(session_id: str) -> str:
    return f"{COPILOT_SESSION_PREFIX}{session_id}"


def node_id_for(tool_name: str) -> str:
    return f"{GATE_NODE_PREFIX}{tool_name}"


def review_id_for(
    session_id: str, user_id: str, tool_name: str, args: dict[str, Any]
) -> str:
    """Bind an approval to this call, in this session, for this user.

    Session and user are inside the hash because ``get_or_create_human_review``
    upserts on ``nodeExecId`` alone, with no ``userId`` in the where clause —
    two sessions issuing ``write_workspace_file(filename="report.md")`` would
    otherwise share one row, and across users the second caller would upsert
    onto the first's row and then be unable to read it back, wedging the gate.
    """
    canonical = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.sha256(
        "\x00".join((session_id, user_id, tool_name, canonical)).encode()
    ).hexdigest()[:16]
    return f"{node_id_for(tool_name)}{COPILOT_NODE_EXEC_ID_SEPARATOR}{digest}"


def review_payload(
    tool_name: str,
    args: dict[str, Any],
    subject: "GateSubject | None" = None,
    *,
    reason: str = "",
    reason_kind: ReasonKind = "mode",
    mode: str | None = None,
    tool_call_id: str = "",
    turn: int = 0,
) -> dict[str, Any]:
    """The subject is kept as decided when the card opened: what the user saw,
    not a recomputation over a tree that may have moved since."""
    # A block's card lists the block's own inputs, each clipped on its own.
    if subject is not None and subject.key.startswith("block:"):
        block_input = args.get("input")
        args = block_input if isinstance(block_input, dict) else {}
    redacted = _redact_secret_keys(args)
    # Per value, never the whole blob: a long first argument must not push
    # the one that matters off the card while the approval still binds it.
    per_value = max(200, _MAX_ARG_CHARS // max(1, len(redacted)))
    shown = {key: _clip(value, per_value) for key, value in redacted.items()}
    return GateReviewPayload(
        tool=tool_name,
        arguments=shown,
        clipped=[key for key in shown if shown[key] is not redacted[key]],
        fields=_field_labels(tool_name, shown),
        tool_call_id=tool_call_id,
        turn=turn,
        mode=mode,
        subject=_payload_subject(tool_name, subject),
        reason=" ".join(reason.split())[:300],
        reason_kind=reason_kind,
        chat_rules_allowed=["allow", "judge"] if subject is not None else [],
        headline=(
            Headline(ask="Run", object=subject.name)
            if subject is not None
            else headline_for(tool_name, args)
        ),
    ).model_dump()


async def find_decision(
    review_id: str, user_id: str, session_id: str
) -> ReviewStatus | None:
    """Status of this exact approval, or None if there isn't one.

    Deliberately does NOT consult ``check_approval``: that also matches the
    ``auto_approve_{graph_exec_id}_{node_id}`` records, which ignore arguments
    entirely and never expire, so one "auto-approve future" toggle on one
    ``bash_exec`` card would clear every later ``bash_exec`` in the session —
    including ones an injected page dictates, and before the taint rule is
    ever reached.
    """
    review = await find_review(review_id, user_id, session_id)
    return review.status if review else None


async def find_review(
    review_id: str, user_id: str, session_id: str
) -> PendingHumanReviewModel | None:
    """This session's row, or None; an approval past its TTL is burnt and None."""
    try:
        reviews = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    except Exception:
        logger.warning(f"Gate could not read review {review_id}", exc_info=True)
        return None
    review = reviews.get(review_id)
    if review is None or review.graph_exec_id != session_exec_id(session_id):
        return None
    approved_at = review.reviewed_at or review.updated_at or review.created_at
    if (
        review.status == ReviewStatus.APPROVED
        and datetime.now(UTC) - approved_at > APPROVAL_TTL
    ):
        await consume(review_id, user_id)
        return None
    return review


async def consume(review_id: str, user_id: str) -> bool:
    """Burn a used approval. The delete IS the mutex.

    Parallel dispatch is deliberate (every MCP tool is ``readOnlyHint=True``),
    so two identical calls can both read APPROVED; only the one whose delete
    removes a row may proceed.
    """
    try:
        return await review_db().delete_review_by_node_exec_id(review_id, user_id) == 1
    except Exception:
        logger.warning(f"Gate could not consume review {review_id}", exc_info=True)
        return False


async def open_review(
    review_id: str,
    user_id: str,
    session: ChatSession,
    tool_name: str,
    args: dict[str, Any],
    reason: str,
    subject: "GateSubject | None" = None,
    reason_kind: ReasonKind = "mode",
    tool_call_id: str = "",
) -> bool:
    """Park the call for approval. False means nothing was recorded."""
    payload = review_payload(
        tool_name,
        args,
        subject,
        reason=reason,
        reason_kind=reason_kind,
        mode=session.metadata.autopilot_mode or DEFAULT_MODE,
        tool_call_id=tool_call_id,
        turn=turn_of(session),
    )
    return await open_review_row(
        review_id, user_id, session, payload, payload_headline(payload)
    )


async def open_review_row(
    review_id: str,
    user_id: str,
    session: ChatSession,
    payload: dict[str, Any],
    message: str,
) -> bool:
    try:
        await review_db().get_or_create_human_review(
            user_id=user_id,
            node_exec_id=review_id,
            graph_exec_id=session_exec_id(session.session_id),
            graph_id=session_exec_id(session.session_id),
            graph_version=1,
            input_data=payload,
            message=message,
            editable=False,
            organization_id=session.organization_id,
            team_id=session.team_id,
        )
        return True
    except Exception:
        logger.warning(
            f"Gate could not open review {review_id} in session "
            f"{session.session_id}",
            exc_info=True,
        )
        return False


def turn_of(session: ChatSession) -> int:
    return sum(1 for m in session.messages if m.role == "user")


def payload_headline(payload: dict[str, Any]) -> str:
    return Headline.model_validate(payload["headline"]).text


def _payload_subject(tool_name: str, subject: "GateSubject | None") -> Subject:
    if subject is None:
        return Subject(
            key=tool_name, name=_label(tool_name), effect=effect_for(tool_name).value
        )
    # ``block:<id>`` or ``workflow:<graph id>``.
    kind, _, ident = subject.key.partition(":")
    return Subject(
        kind=kind or "tool",
        key=subject.key,
        name=subject.name,
        effect=subject.effect.value,
        irreversible=subject.irreversible,
        block_id=ident if kind == "block" else None,
    )


def _label(tool_name: str) -> str:
    label = tool_name.replace("_", " ")
    return label[:1].upper() + label[1:]


def _field_labels(tool_name: str, shown: dict[str, Any]) -> list[FieldLabel]:
    """Labels and order from the tool's own input schema, required first."""
    from backend.copilot.tools import get_tool  # imports the gate

    tool = get_tool(tool_name)
    schema = tool.parameters if tool else {}
    props = schema.get("properties") or {}
    required = [k for k in schema.get("required") or [] if k in props]
    order = required + [k for k in props if k not in required]
    order += [k for k in shown if k not in order]
    return [
        FieldLabel(key=key, label=(props.get(key) or {}).get("title") or _humanize(key))
        for key in order
        if key in shown
    ]


def _humanize(key: str) -> str:
    words = key.removesuffix("_id").replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _clip(value: Any, limit: int) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…"
