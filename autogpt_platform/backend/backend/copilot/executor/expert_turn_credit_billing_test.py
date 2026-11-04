"""An expert's AutoPilot block turn, billed to credits, held for a fresh hire.

The turn runs as the executor runs it, with the real credit billing
(``bill_to_credits`` and ``credit_billing``) on an in-memory balance, and an
engine that records what the turn cost, as the SDK engine's cost tracking
does. The executor checks the expert, holds the turn while the hire's setup
runs, checks the expert again, and only then bills the balance and starts the
engine: a turn whose expert changed during the wait is refused before either
and costs nothing.
"""

import logging
import threading
from contextlib import ExitStack
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from backend.api.features.experts.models import CheckedExpert
from backend.copilot.config import ChatConfig
from backend.copilot.constants import AUTOPILOT_BLOCK_TOOL_NAME
from backend.copilot.credit_billing import record_credit_billed_cost
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.expert_check_test import CHECK
from backend.copilot.expert_context import (
    EXPERT_SESSION_MISSING_MESSAGE,
    EXPERT_SESSION_TEMPORARY_MESSAGE,
    ExpertSessionUnavailableError,
    build_expert_identity_suffix,
)
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamTextDelta
from backend.copilot.turn_pause import TurnPause
from backend.util.exceptions import ExpertNotFoundError

_PROCESSOR = "backend.copilot.executor.processor"
OUT_OF_TODAY = TurnPause(
    reason="daily",
    resets_at=datetime(2026, 11, 4, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)
# 2.34 credits: the turn pays 2 as the cost lands and the rest, rounded up to
# 1, when it ends.
TURN_COST_MICRODOLLARS = 23_400
STARTING_BALANCE = 500
INSTALLING = CHECK.model_copy(
    update={"expert": CHECK.expert.model_copy(update={"setup_status": "installing"})}
)
SET_UP = CHECK.model_copy(
    update={"expert": CHECK.expert.model_copy(update={"name": "Maria after setup"})}
)
MOVED = SET_UP.model_copy(update={"organization_id": "another-org"})
WAITS = [
    pytest.param(["installing", "ready"], 1.0, id="set-up-during-the-wait"),
    pytest.param(["installing"], 0.0, id="still-installing-at-the-deadline"),
]


class _Balance:
    """The owner's credit balance, kept as the credit model keeps it, and
    every debit the turn takes from it."""

    def __init__(self, events: list[str]) -> None:
        self.credits = STARTING_BALANCE
        self.events = events

    async def get_personal_org_owner(self, organization_id: str) -> str | None:
        return "user-1" if organization_id == CHECK.organization_id else None

    async def get_credits(self, user_id: str) -> int:
        return self.credits

    async def spend_credits(self, *, user_id: str, cost: int, metadata) -> int:
        assert user_id == "user-1"
        self.credits -= cost
        self.events.append(f"charged {cost}")
        return self.credits

    async def get_org_credits(self, **_kwargs) -> int:
        raise AssertionError("a personal organization's turn bills its owner")

    async def spend_org_credits(self, **_kwargs) -> int:
        raise AssertionError("a personal organization's turn bills its owner")


@pytest.fixture(autouse=True)
def _credits_pay_block_turns(mocker):
    mocker.patch("backend.copilot.credit_billing.settings.config.enable_credit", True)
    mocker.patch.object(
        ChatConfig, "thinking_available", new_callable=PropertyMock, return_value=True
    )


@pytest.fixture(autouse=True)
def _no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


async def _run_block_turn(
    checks: list[CheckedExpert | Exception],
    setup_statuses: list[str],
    wait_seconds: float,
) -> tuple[_Balance, list[str], list[str], ExpertSessionUnavailableError | None]:
    """Run one block turn whose allowance is used up, so the executor bills it
    to credits. ``checks`` are what each read of the expert finds, in order.
    Returns the balance, what happened in order, the identity the engine was
    given, and the refusal if the turn was refused."""
    events: list[str] = []
    identities: list[str] = []
    balance = _Balance(events)
    found = iter(checks)

    async def check_private_expert(user_id: str, expert_id: str) -> CheckedExpert:
        assert (user_id, expert_id) == ("user-1", "expert-1")
        events.append("checked again" if "checked" in events else "checked")
        check = next(found)
        if isinstance(check, Exception):
            raise check
        return check

    store = MagicMock()
    store.check_private_expert = AsyncMock(side_effect=check_private_expert)
    store.expert_setup_status = AsyncMock(side_effect=list(setup_statuses))

    async def engine(*, session: ChatSession, user_id: str, **_kwargs):
        events.append("engine started")
        identities.append(
            await build_expert_identity_suffix(
                user_id,
                session.expert_id,
                organization_id=session.organization_id,
                team_id=session.team_id,
            )
        )
        await record_credit_billed_cost(user_id, TURN_COST_MICRODOLLARS)
        yield StreamTextDelta(id="t", delta="done")

    session = ChatSession.new(
        "user-1",
        dry_run=False,
        organization_id=CHECK.organization_id,
        team_id=CHECK.team_id,
        expert_id="expert-1",
    )
    session.session_id = "sess-1"
    entry = CoPilotExecutionEntry(
        session_id="sess-1",
        turn_id="turn-1",
        user_id="user-1",
        message="Run the weekly report",
        tool_name=AUTOPILOT_BLOCK_TOOL_NAME,
    )
    refused = None
    with ExitStack() as stack:
        for target, patch_args in (
            (f"{_PROCESSOR}.ChatConfig", {"return_value": MagicMock(test_mode=True)}),
            (f"{_PROCESSOR}.stream_chat_completion_dummy", {"new": engine}),
            (
                f"{_PROCESSOR}.stream_registry.stream_and_publish",
                {"side_effect": lambda **kwargs: kwargs["stream"]},
            ),
            (f"{_PROCESSOR}.stream_registry.publish_chunk", {"new": AsyncMock()}),
            (
                f"{_PROCESSOR}.stream_registry.mark_session_completed",
                {"new": AsyncMock()},
            ),
            (
                f"{_PROCESSOR}.close_browser_daemon",
                {"new": AsyncMock(return_value=False)},
            ),
            (
                "backend.copilot.model.get_chat_session",
                {"new": AsyncMock(return_value=session)},
            ),
            ("backend.data.db_accessors.experts_db", {"return_value": store}),
            ("backend.data.db_accessors.expert_check_db", {"return_value": store}),
            ("backend.copilot.expert_context.experts_db", {"return_value": store}),
            (f"{_PROCESSOR}.EXPERT_SETUP_WAIT_SECONDS", {"new": wait_seconds}),
            (f"{_PROCESSOR}.EXPERT_SETUP_POLL_SECONDS", {"new": 0}),
            (
                f"{_PROCESSOR}.refusal_for",
                {"new": AsyncMock(return_value=OUT_OF_TODAY)},
            ),
            ("backend.copilot.credit_billing.credit_db", {"return_value": balance}),
        ):
            stack.enter_context(patch(target, **patch_args))
        try:
            await CoPilotProcessor()._execute_async(
                entry,
                threading.Event(),
                MagicMock(),
                CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
            )
        except ExpertSessionUnavailableError as error:
            refused = error
    return balance, events, identities, refused


@pytest.mark.asyncio
async def test_a_ready_experts_turn_is_checked_once_and_charged_once():
    balance, events, identities, refused = await _run_block_turn([CHECK], [], 1.0)

    assert refused is None
    assert events == ["checked", "engine started", "charged 2", "charged 1"]
    assert balance.credits == STARTING_BALANCE - 3
    assert len(identities) == 1
    assert "For this session you are Maria" in identities[0]


@pytest.mark.parametrize(("setup_statuses", "wait_seconds"), WAITS)
@pytest.mark.asyncio
async def test_a_held_turn_is_charged_once_after_its_expert_is_checked_again(
    setup_statuses: list[str], wait_seconds: float
):
    balance, events, identities, refused = await _run_block_turn(
        [INSTALLING, SET_UP], setup_statuses, wait_seconds
    )

    assert refused is None
    assert events == [
        "checked",
        "checked again",
        "engine started",
        "charged 2",
        "charged 1",
    ]
    assert balance.credits == STARTING_BALANCE - 3
    assert len(identities) == 1
    assert "For this session you are Maria after setup" in identities[0]


@pytest.mark.parametrize(
    ("after_the_wait", "message"),
    [
        pytest.param(
            ExpertNotFoundError("expert-1"),
            EXPERT_SESSION_MISSING_MESSAGE,
            id="archived",
        ),
        pytest.param(MOVED, EXPERT_SESSION_TEMPORARY_MESSAGE, id="moved"),
    ],
)
@pytest.mark.parametrize(("setup_statuses", "wait_seconds"), WAITS)
@pytest.mark.asyncio
async def test_a_held_turn_whose_expert_changed_is_refused_and_costs_nothing(
    after_the_wait: CheckedExpert | Exception,
    message: str,
    setup_statuses: list[str],
    wait_seconds: float,
):
    balance, events, identities, refused = await _run_block_turn(
        [INSTALLING, after_the_wait], setup_statuses, wait_seconds
    )

    assert refused is not None and str(refused) == message
    assert events == ["checked", "checked again"]
    assert balance.credits == STARTING_BALANCE
    assert identities == []
