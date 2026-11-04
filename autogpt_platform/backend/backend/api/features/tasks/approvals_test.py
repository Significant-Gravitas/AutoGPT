from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.tasks import approvals, approvals_answer
from backend.api.features.tasks.models import (
    CardApproval,
    CardDecision,
    Contract,
    Task,
    TaskStatus,
)
from backend.copilot.gate.card_approval import _DECISION

DEVIN = CardApproval(confirmation_id="c-devin", kind="hire", name="Devin", role="Dev")
MAYA = CardApproval(confirmation_id="c-maya", kind="raise", name="Maya")


def test_a_reproposed_card_replaces_the_one_for_the_same_expert():
    first, _ = approvals.add_approval(Contract(goal="g"), DEVIN)
    again = DEVIN.model_copy(update={"confirmation_id": "c-devin-2"})

    updated, patches = approvals.add_approval(first, again)

    assert [a.confirmation_id for a in updated.approvals] == ["c-devin-2"]
    assert patches[0].field == "approvals"


def test_adding_the_same_card_twice_changes_nothing():
    first, _ = approvals.add_approval(Contract(goal="g"), DEVIN)

    _, patches = approvals.add_approval(first, DEVIN)

    assert patches == []


def test_decision_lines_are_the_lines_the_chat_card_sends():
    lines = approvals.decision_lines(
        [DEVIN, MAYA],
        [
            CardDecision(confirmation_id="c-devin", approved=True),
            CardDecision(confirmation_id="c-maya", approved=False),
            CardDecision(confirmation_id="c-unknown", approved=True),
        ],
    )

    assert lines == [
        "Approved: hire Devin (confirmation_id: c-devin).",
        "Not approved: do not create Maya, discard that proposal "
        "(confirmation_id: c-maya).",
    ]
    # The gate's own pattern reads every one of them.
    assert all(_DECISION.match(line) for line in lines)


def test_answer_text_names_who_was_approved_and_declined():
    text = approvals.answer_text(
        [DEVIN, MAYA],
        [
            CardDecision(confirmation_id="c-devin", approved=True),
            CardDecision(confirmation_id="c-maya", approved=False),
        ],
    )

    assert text == "Approved Devin; Declined Maya"


def _task(**overrides) -> Task:
    now = datetime.now(UTC)
    fields = {
        "id": "t-1",
        "number": 3,
        "key": "T-3",
        "title": "Raise two experts",
        "status": TaskStatus.BLOCKED,
        "priority": "none",
        "work_session_id": "s-work",
        "approvals": [DEVIN, MAYA],
        "created_at": now,
        "updated_at": now,
    }
    return Task.model_validate({**fields, **overrides})


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch):
    db = MagicMock()
    db.get_task = AsyncMock(return_value=_task())
    db.answer_input = AsyncMock()
    contract_db = MagicMock()
    contract_db.drop_approvals = AsyncMock(return_value=MagicMock(patches=[]))
    record = AsyncMock()
    monkeypatch.setattr(approvals_answer, "db", db)
    monkeypatch.setattr(approvals_answer, "contract_db", contract_db)
    monkeypatch.setattr(approvals_answer, "record_card_decisions", record)
    monkeypatch.setattr(approvals_answer, "publish_contract_update", AsyncMock())
    return db, contract_db, record


@pytest.mark.asyncio
async def test_approving_on_the_board_records_the_approval_in_the_work_chat(stubs):
    db, contract_db, record = stubs

    message = await approvals_answer.answer_cards(
        "u-1",
        "t-1",
        [
            CardDecision(confirmation_id="c-devin", approved=True),
            CardDecision(confirmation_id="c-maya", approved=True),
        ],
    )

    record.assert_awaited_once_with(
        "u-1",
        "s-work",
        "Approved: hire Devin (confirmation_id: c-devin).\n"
        "Approved: create Maya (confirmation_id: c-maya).",
    )
    contract_db.drop_approvals.assert_awaited_once_with(
        "u-1", "t-1", ["c-devin", "c-maya"]
    )
    assert db.answer_input.await_args.args[2] == "Approved Devin and Maya"
    # The worker reads the ids it needs to confirm from the resume message.
    assert "(confirmation_id: c-devin)." in message
    assert "(confirmation_id: c-maya)." in message


@pytest.mark.asyncio
async def test_a_card_the_task_no_longer_holds_is_refused(stubs):
    _, _, record = stubs

    with pytest.raises(ValueError):
        await approvals_answer.answer_cards(
            "u-1", "t-1", [CardDecision(confirmation_id="c-gone", approved=True)]
        )

    record.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_task_without_a_work_chat_is_refused(stubs):
    db, _, record = stubs
    db.get_task.return_value = _task(work_session_id=None)

    with pytest.raises(ValueError):
        await approvals_answer.answer_cards(
            "u-1", "t-1", [CardDecision(confirmation_id="c-devin", approved=True)]
        )

    record.assert_not_awaited()
