"""What the platform charged, captured where ``persist_and_record_usage`` writes.

Each write lands in one of three places: the user's daily and weekly spend
counters, the turn tree's budget, and a ``PlatformCostLog`` row. A turn billed
to credits debits the credit balance instead of the spend counters. The ledger
replaces exactly those writes, and stands in for the credit balance, so every
decision about what to charge still runs for real, and files each write under
the turn that made it.
"""

import asyncio

import pytest
from pydantic import BaseModel

from backend.copilot import credit_billing, token_tracking
from backend.copilot.baseline import service as baseline_service
from backend.copilot.sdk import service as sdk_service
from backend.copilot.tree import TurnEnvelope
from backend.data.credit import UsageTransactionMetadata
from backend.data.platform_cost import PlatformCostEntry
from backend.util.exceptions import InsufficientBalanceError


class SpendCharge(BaseModel):
    turn: int
    user_id: str
    microdollars: int
    skip_daily: bool


class TreeCharge(BaseModel):
    turn: int
    tree_id: str
    microdollars: int


class CostLogRow(BaseModel):
    turn: int
    entry: PlatformCostEntry


class CreditDebit(BaseModel):
    turn: int
    user_id: str
    credits: int
    metadata: UsageTransactionMetadata


class PlannedDrain(BaseModel):
    """Other runs spending the credit balance while one turn runs."""

    turn: int
    # What they leave once the turn has read the balance.
    left: int


class CreditBilling(BaseModel):
    """The plan's allowance is used up, and each turn, the AutoPilot block's,
    is billed to this credit balance instead."""

    # Over the per-turn spend floor, so each turn runs on it, and far over a
    # whole chat's cost here, so it never runs short.
    balance: int = 500
    drain: PlannedDrain | None = None

    def available(self, turn: int) -> int | None:
        """What the balance leaves ``turn`` to pay; None where it covers the
        turn's whole cost."""
        if self.drain is None or self.drain.turn != turn:
            return None
        return self.drain.left


class PlatformLedger:
    def __init__(self) -> None:
        self.turn = 0
        self.spend: list[SpendCharge] = []
        self.tree: list[TreeCharge] = []
        self.cost_log: list[CostLogRow] = []
        self.debits: list[CreditDebit] = []
        self.balance = 0
        self.drain: PlannedDrain | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(token_tracking, "record_cost_usage", self.record_spend)
        monkeypatch.setattr(token_tracking, "charge_turn", self._charge_tree)
        monkeypatch.setattr(token_tracking, "platform_cost_db", lambda: self)
        monkeypatch.setattr(credit_billing, "credit_db", lambda: self)

    def fund(self, credits: CreditBilling) -> None:
        self.balance = credits.balance
        self.drain = credits.drain

    def spend_in(self, turn: int) -> list[SpendCharge]:
        return [charge for charge in self.spend if charge.turn == turn]

    def tree_in(self, turn: int) -> list[TreeCharge]:
        return [charge for charge in self.tree if charge.turn == turn]

    def rows_in(self, turn: int) -> list[PlatformCostEntry]:
        return [row.entry for row in self.cost_log if row.turn == turn]

    def debits_in(self, turn: int) -> list[CreditDebit]:
        return [debit for debit in self.debits if debit.turn == turn]

    async def log_platform_cost(self, entry: PlatformCostEntry) -> None:
        self.cost_log.append(CostLogRow(turn=self.turn, entry=entry))

    async def get_credits(self, user_id: str) -> int:
        balance = self.balance
        if self.drain is not None and self.drain.turn == self.turn:
            # Other runs spend it down as soon as the turn has read it.
            self.balance = min(self.balance, self.drain.left)
        return balance

    async def spend_credits(
        self, *, user_id: str, cost: int, metadata: UsageTransactionMetadata
    ) -> int:
        if cost > self.balance:
            # Refused whole, as the credit model refuses a debit it can't cover.
            raise InsufficientBalanceError(
                "Insufficient balance", user_id, self.balance, cost
            )
        self.balance -= cost
        self.debits.append(
            CreditDebit(
                turn=self.turn, user_id=user_id, credits=cost, metadata=metadata
            )
        )
        return self.balance

    async def settle(self) -> None:
        """Wait for the writes a turn defers past its own end.

        The OpenRouter reconcile charges from a background task, and every
        cost-log row is written by one; both must land under the turn that
        started them.
        """
        while pending := _deferred_tasks():
            await asyncio.gather(*pending)

    async def record_spend(
        self, user_id: str, cost_microdollars: int, *, skip_daily: bool = False
    ) -> None:
        self.spend.append(
            SpendCharge(
                turn=self.turn,
                user_id=user_id,
                microdollars=cost_microdollars,
                skip_daily=skip_daily,
            )
        )

    async def _charge_tree(self, envelope: TurnEnvelope, microdollars: int) -> None:
        self.tree.append(
            TreeCharge(
                turn=self.turn, tree_id=envelope.tree_id, microdollars=microdollars
            )
        )


def _deferred_tasks() -> list[asyncio.Task]:
    with token_tracking._pending_log_tasks_lock:
        logs = list(token_tracking._pending_log_tasks)
    loop = asyncio.get_running_loop()
    tasks = [
        *logs,
        *sdk_service._background_tasks,
        *baseline_service._background_tasks,
    ]
    return [task for task in tasks if task.get_loop() is loop and not task.done()]
