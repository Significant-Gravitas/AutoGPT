"""The seam for non-registry MCP tools refuses whenever the gate cannot decide."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate import Decision
from backend.copilot.gate.mcp_seam import (
    Verdict,
    approved_for,
    gate_non_registry_tool,
    running_under,
)
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.subject import Subject
from backend.copilot.model import ChatSession

_CHECK = "backend.copilot.gate.mcp_seam.check_action"


async def test_a_crashing_gate_refuses_rather_than_runs():
    session = ChatSession.new(user_id="u1", dry_run=False)
    with patch(_CHECK, new=AsyncMock(side_effect=RuntimeError("gate down"))):
        result, decision = await gate_non_registry_tool(
            "write_file", {"path": "a"}, "u1", session
        )
    assert result is not None
    assert result["isError"] is True
    assert decision is None


async def test_a_parked_call_mounts_the_approval_card():
    session = ChatSession.new(user_id="u1", dry_run=False)
    parked = Decision(allowed=False, reason="needs you", review_id="r1")
    with patch(_CHECK, new=AsyncMock(return_value=parked)):
        result, decision = await gate_non_registry_tool(
            "write_file", {"path": "a"}, "u1", session
        )
    assert result is not None and decision is None
    payload = json.loads(result["content"][0]["text"])
    assert payload["type"] == "approval_required"
    assert payload["review_id"] == "r1"


async def test_an_allowed_call_proceeds_with_the_gates_decision():
    session = ChatSession.new(user_id="u1", dry_run=False)
    allowed = Decision(allowed=True, decided=True)
    with patch(_CHECK, new=AsyncMock(return_value=allowed)):
        result, verdict = await gate_non_registry_tool(
            "read_file", {"p": 1}, "u1", session
        )
    assert result is None
    assert verdict == Verdict(decision=allowed)


_DISK = Subject(key="disk:a", name="Disk", effect=Effect.EXTERNAL)


async def test_the_verdict_names_the_subject_the_gate_decided_on():
    session = ChatSession.new(user_id="u1", dry_run=False)

    async def check(*_args, subject_of=None, **_kwargs):
        assert subject_of is not None
        assert await subject_of() == _DISK
        return Decision(allowed=True, decided=True)

    async def subject_of():
        return _DISK

    with patch(_CHECK, new=check):
        _, verdict = await gate_non_registry_tool(
            "x", {}, "u1", session, subject_of=subject_of
        )
    assert verdict is not None and verdict.subject == _DISK


@pytest.mark.parametrize(
    "verdict, approved",
    [
        # The gate is off or cannot ask: it allows without deciding.
        (Verdict(decision=Decision(allowed=True), subject=_DISK), False),
        # The mode or a rule ruled on this very subject.
        (Verdict(decision=Decision(allowed=True, decided=True), subject=_DISK), True),
        # It ruled on another subject.
        (
            Verdict(
                decision=Decision(allowed=True, decided=True),
                subject=_DISK.model_copy(update={"key": "disk:b"}),
            ),
            False,
        ),
        (Verdict(decision=Decision(allowed=True, decided=True)), False),
        # The user approved exactly this call on its card.
        (
            Verdict(decision=Decision(allowed=True, approved=True, decided=True)),
            True,
        ),
    ],
)
def test_a_handler_reads_the_verdict_on_its_own_call_only(verdict, approved):
    assert approved_for("disk:a") is False
    with running_under(verdict):
        assert approved_for("disk:a") is approved
    assert approved_for("disk:a") is False
