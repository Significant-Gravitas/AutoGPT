"""What the platform charged, captured where ``persist_and_record_usage`` writes.

Each write lands in one of three places: the user's daily and weekly spend
counters, the turn tree's budget, and a ``PlatformCostLog`` row. The ledger
replaces exactly those three writes, so every decision about what to charge
still runs for real, and files each write under the turn that made it.
"""

import asyncio

import pytest
from pydantic import BaseModel

from backend.copilot import token_tracking
from backend.copilot.baseline import service as baseline_service
from backend.copilot.sdk import service as sdk_service
from backend.copilot.tree import TurnEnvelope
from backend.data.platform_cost import PlatformCostEntry


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


class PlatformLedger:
    def __init__(self) -> None:
        self.turn = 0
        self.spend: list[SpendCharge] = []
        self.tree: list[TreeCharge] = []
        self.cost_log: list[CostLogRow] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(token_tracking, "record_cost_usage", self._record_spend)
        monkeypatch.setattr(token_tracking, "charge_turn", self._charge_tree)
        monkeypatch.setattr(token_tracking, "platform_cost_db", lambda: self)

    def spend_in(self, turn: int) -> list[SpendCharge]:
        return [charge for charge in self.spend if charge.turn == turn]

    def tree_in(self, turn: int) -> list[TreeCharge]:
        return [charge for charge in self.tree if charge.turn == turn]

    def rows_in(self, turn: int) -> list[PlatformCostEntry]:
        return [row.entry for row in self.cost_log if row.turn == turn]

    async def log_platform_cost(self, entry: PlatformCostEntry) -> None:
        self.cost_log.append(CostLogRow(turn=self.turn, entry=entry))

    async def settle(self) -> None:
        """Wait for the writes a turn defers past its own end.

        The OpenRouter reconcile charges from a background task, and every
        cost-log row is written by one; both must land under the turn that
        started them.
        """
        while pending := _deferred_tasks():
            await asyncio.gather(*pending)

    async def _record_spend(
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
