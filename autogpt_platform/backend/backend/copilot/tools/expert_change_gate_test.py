"""A hire or raise asks the user once: the preview never holds, and the card's
own Approve answers the confirm.

Driven through ``BaseTool.execute`` with the gate on, as both engines call it.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.gate.classifier import Judgement
from backend.copilot.gate.headline import Headline
from backend.copilot.model import (
    AutopilotMode,
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
)

from . import hire_expert, raise_expert
from .base import BaseTool
from .confirm_expert_change import ConfirmExpertChangeTool
from .expert_change_test import _CHARTER, _USER, _env
from .hire_expert import HireExpertTool
from .raise_expert import RaiseExpertTool
from .update_expert import UpdateExpertTool

_GATE = "backend.copilot.gate"
FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/components/ToolChain/__tests__"
    / "gatedProposal.json"
)
_ID = "5f0c2a3e-7b1d-4c9e-8a6f-0d2b4e6c8a10"
# What the card sends: ``decisionLine`` in ExpertCards.tsx, pinned by its test.
_APPROVED = f"Approved: create Otto (confirmation_id: {_ID})."
_IN_A_GROUP = (
    "Not approved: do not create Ada, discard that proposal "
    f"(confirmation_id: c-other).\n{_APPROVED}"
)
_DECLINED = (
    "Not approved: do not create Otto, discard that proposal "
    f"(confirmation_id: {_ID})."
)
_FIXED_UUID = SimpleNamespace(uuid4=lambda: uuid.UUID(_ID))
_PREVIEWS: list[tuple[BaseTool, dict[str, Any]]] = [
    (HireExpertTool(), {"template_id": "tpl-scout"}),
    (RaiseExpertTool(), _CHARTER),
    (UpdateExpertTool(), {"expert_id": "exp-2", "name": "Nick"}),
]


@pytest.fixture
def gate():
    """The gate on, no approval on record; the supervisor holds whatever it sees."""
    store = SimpleNamespace(
        find_review=AsyncMock(return_value=None),
        open_review=AsyncMock(return_value=Headline(ask="Confirm the team change")),
        supervise=AsyncMock(
            return_value=Judgement(allowed=False, reason="Check with the user.")
        ),
    )
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.find_review", store.find_review),
        patch(f"{_GATE}.review_store.open_review", store.open_review),
        patch(f"{_GATE}.held.remember", AsyncMock(return_value=True)),
        patch(f"{_GATE}.chat_rules.rule_for", AsyncMock(return_value=None)),
        patch(f"{_GATE}.supervise", store.supervise),
        patch(f"{_GATE}.reads.release_held_read", AsyncMock(return_value=None)),
        patch(f"{_GATE}.reads.screen_read", AsyncMock(return_value=None)),
        patch.object(hire_expert, "uuid", _FIXED_UUID),
        patch.object(raise_expert, "uuid", _FIXED_UUID),
    ):
        yield store


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
@pytest.mark.parametrize(
    "tool, args", _PREVIEWS, ids=[tool.name for tool, _ in _PREVIEWS]
)
async def test_a_preview_puts_its_card_up_without_a_gate_card(gate, mode, tool, args):
    with _env():
        output = _output(await tool.execute(_USER, _session(mode), "call-1", **args))

    assert output["type"] == "expert_change_proposed"
    assert output["confirmation_id"]
    gate.open_review.assert_not_awaited()


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
@pytest.mark.parametrize(
    "preview, args, reply, created",
    [
        (
            HireExpertTool(),
            {"template_id": "tpl-scout"},
            f"Approved: hire Scout (confirmation_id: {_ID}).",
            lambda db: db.hire_expert,
        ),
        (RaiseExpertTool(), _CHARTER, _APPROVED, lambda db: db.create_raised_expert),
        # Send decisions puts one line per proposal in a single message.
        (RaiseExpertTool(), _CHARTER, _IN_A_GROUP, lambda db: db.create_raised_expert),
    ],
    ids=["hire", "raise", "raise-in-a-group"],
)
async def test_one_approval_on_the_card_creates_the_teammate(
    gate,
    mode,
    preview: BaseTool,
    args: dict[str, Any],
    reply: str,
    created: Callable[[MagicMock], AsyncMock],
):
    session = _session(mode)
    with _env() as db:
        await preview.execute(_USER, session, "call-1", **args)
        _reply(session, reply)
        output = _output(
            await ConfirmExpertChangeTool().execute(
                _USER, session, "call-2", confirmation_id=_ID
            )
        )

    assert output["type"] == "expert_change_applied"
    created(db).assert_awaited_once()
    gate.open_review.assert_not_awaited()
    gate.supervise.assert_not_awaited()


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
@pytest.mark.parametrize(
    "reply, metadata",
    [
        ("yes", None),
        (_DECLINED, None),
        (_APPROVED.replace(_ID, "c-other"), None),
        # A held call's late result is a user row no person typed.
        (_APPROVED, {"held_call": {"review_id": "r-1"}}),
    ],
    ids=["typed-yes", "declined", "another-proposal", "gate-written-row"],
)
async def test_a_confirm_the_card_did_not_approve_still_goes_to_the_gate(
    gate, mode, reply, metadata
):
    session = _session(mode)
    with _env() as db:
        await RaiseExpertTool().execute(_USER, session, "call-1", **_CHARTER)
        _reply(session, reply, metadata)
        result = await ConfirmExpertChangeTool().execute(
            _USER, session, "call-2", confirmation_id=_ID
        )

    assert _output(result)["type"] == "approval_required"
    db.create_raised_expert.assert_not_awaited()
    gate.open_review.assert_awaited_once()


async def test_the_frontend_proposal_fixture_is_what_a_gated_preview_returns(gate):
    """The chat renders this output in ExpertCards.test.tsx; a preview that
    holds again returns a gate refusal here instead, and no Approve button."""
    with _env():
        result = await RaiseExpertTool().execute(
            _USER, _session("ask_first"), "call-1", **_CHARTER
        )
    output = _output(result)
    if os.environ.get("UPDATE_CARD_FIXTURE"):
        FIXTURE.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    assert (
        json.loads(FIXTURE.read_text()) == output
    ), "the gated-proposal fixture is stale; rerun with UPDATE_CARD_FIXTURE=1"


def _session(mode: AutopilotMode) -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id=_USER,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=[ChatMessage(role="user", content="Add a teammate", sequence=0)],
    )


def _reply(session: ChatSession, content: str, metadata: dict | None = None) -> None:
    session.messages.append(
        ChatMessage(
            role="user",
            content=content,
            sequence=len(session.messages),
            metadata=metadata,
        )
    )


def _output(result) -> dict[str, Any]:
    return json.loads(result.output)
