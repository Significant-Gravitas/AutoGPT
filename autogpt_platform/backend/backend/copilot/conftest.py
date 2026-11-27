"""Fixtures shared by every copilot test."""

import pytest

from backend.copilot.credit_billing import CreditBilledTurn
from backend.copilot.credit_turn_holds import LedgerHolds
from backend.copilot.turn_pause import TurnPause


@pytest.fixture(autouse=True)
def _no_bans(mocker):
    """No owner is banned or closed unless a test says so. Without this the
    lookup goes to a DatabaseManager these tests don't run, and waits out its
    retries."""
    for module in (
        "backend.copilot.turn_queue",
        "backend.copilot.briefing.generate",
    ):
        mocker.patch(
            f"{module}.is_user_refused", new=mocker.AsyncMock(return_value=False)
        )
    mocker.patch(
        "backend.copilot.executor.processor.account_refusal",
        new=mocker.AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def _no_cli_prelaunch(mocker):
    """No turn spawns a real CLI ahead of its query unless a test says so: a
    mocked ``ClaudeSDKClient`` would take the process and never stop it.
    Tests of the prelaunch patch ``service.CLIPrelaunch`` back themselves."""
    mocker.patch("backend.copilot.sdk.service.CLIPrelaunch", _NoCLIPrelaunch)


class _NoCLIPrelaunch:
    @staticmethod
    async def start(options, log_prefix):
        return None


class OwnHolds:
    """``LedgerHolds`` for suites that bill turns but test something else:
    each turn alone on its ledger, holding nothing on Redis."""

    made: list["OwnHolds"] = []

    def __init__(self, ledger_id: str, lifetime_seconds: int) -> None:
        self.ledger_id = ledger_id
        self.held = 0
        self.taken: list[int] = []
        self.lowered: list[int] = []
        self.released = 0
        OwnHolds.made.append(self)

    async def hold(
        self, turn_id: str, *, read_balance, credits: int, at_least: int = 1
    ) -> int:
        added = min(await read_balance() - self.held, credits)
        if added < at_least:
            return 0
        self.held += added
        self.taken.append(added)
        return added

    async def lower(self, turn_id: str, credits: int) -> None:
        self.held = credits
        self.lowered.append(credits)

    async def release(self, turn_id: str) -> None:
        self.released += 1


@pytest.fixture
def own_ledger_holds(mocker) -> type[OwnHolds]:
    """Credit-billed turns that hold nothing on the shared Redis, so a run
    that ends one without releasing it can't block a later run for 6 h."""
    OwnHolds.made.clear()
    mocker.patch("backend.copilot.credit_billing.LedgerHolds", OwnHolds)
    return OwnHolds


class _LedgerWithRoom(LedgerHolds):
    """A ledger with a fixed number of credits left to hold, off Redis."""

    def __init__(self, room: int) -> None:
        super().__init__("test-ledger", lifetime_seconds=60)
        self.room = room

    async def hold(
        self, turn_id: str, *, read_balance, credits: int, at_least: int = 1
    ) -> int:
        added = min(self.room, credits)
        if added < at_least:
            return 0
        self.room -= added
        return added

    async def lower(self, turn_id: str, credits: int) -> None:
        pass

    async def release(self, turn_id: str) -> None:
        pass


@pytest.fixture
def growing_turn() -> CreditBilledTurn:
    """A credit-billed turn holding $1 of the $5 it may hold, on a ledger with
    $1 more free."""
    return CreditBilledTurn(
        user_id="u-1",
        session_id="s1",
        cap_credits=500,
        refusal=TurnPause(reason="daily", started=False),
        holds=_LedgerWithRoom(room=100),
        turn_id="turn-1",
        held_credits=100,
    )
