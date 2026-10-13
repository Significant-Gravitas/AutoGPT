"""AutoPilot block turns billed to the credit balance once the allowance is out.

A graph running the AutoPilot block shouldn't stop because the plan's AutoPilot
allowance is used up while the account's automation credits could pay for the
turn. So before the executor refuses a block turn for budget (see
``backend.copilot.budget_gate``), it checks the credit balance: if the balance
covers the per-turn spend floor, the turn runs billed to credits instead. Only
the block's turns do, its recovery re-run included. Chat, scheduled follow-ups
and routines, sub-sessions, approval wakes, chat bots and auto-continue are
refused as before.

A credit-billed turn:

- may spend what the balance covered when it started, and never more than the
  per-turn ceiling (``claude_agent_max_budget_usd``). Its budget is a hard cap,
  so its CLI backstop sits exactly on it;
- runs on the SDK engine, the one that enforces a budget;
- records its cost here rather than in the plan's daily and weekly windows;
- is charged once when it ends: the cost it recorded, rounded up to whole
  credits, never past its cap. A turn that ended without a result is charged
  what it recorded too, through the same accounting as any other turn.
"""

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from backend.copilot.config import ChatConfig
from backend.copilot.constants import (
    AUTOPILOT_BLOCK_ID,
    AUTOPILOT_BLOCK_TOOL_NAME,
    PER_TURN_SPEND_FLOOR_USD,
)
from backend.copilot.rate_limit import RateLimitUnavailable
from backend.copilot.tree import MICRODOLLARS_PER_CREDIT
from backend.copilot.turn_budget import BudgetWindow, TurnBudget
from backend.copilot.turn_pause import PauseReason, TurnPause
from backend.data.credit import UsageTransactionMetadata
from backend.data.db_accessors import credit_db
from backend.util.exceptions import InsufficientBalanceError

logger = logging.getLogger(__name__)
config = ChatConfig()

CREDITS_PER_USD = 100
# The OpenRouter cost reconcile retries its lookups for up to ~30s after the
# stream ends (see ``sdk.openrouter_cost``). The charge waits that long for it;
# a cost recorded later still is charged when it lands.
_COST_SETTLE_TIMEOUT_SECONDS = 45.0
_BUDGET_WINDOW: dict[PauseReason, BudgetWindow] = {
    "daily": "daily",
    "weekly": "weekly",
    "trial": "trial",
    "task": "task",
    "steps": "task",
}

_credit_billed_turn: ContextVar["CreditBilledTurn | None"] = ContextVar(
    "credit_billed_turn", default=None
)


class CreditBilledTurn:
    """One AutoPilot block turn running on the credit balance."""

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str,
        cap_credits: int,
        refusal: TurnPause,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        # What the balance covered when the turn started, within the ceiling.
        self.cap_credits = cap_credits
        # The refusal this turn runs instead of: the limit that's used up.
        self.refusal = refusal
        self.spent_microdollars = 0
        self.charged_credits = 0
        self._ended = False
        self._recordings: set[asyncio.Task] = set()
        self._charging = asyncio.Lock()

    @property
    def budget(self) -> TurnBudget:
        """A hard cap at what the balance covers: the turn mustn't overdraw."""
        return TurnBudget(
            usd=self.cap_credits / CREDITS_PER_USD,
            limiting=_BUDGET_WINDOW[self.refusal.reason],
            resets_at=self.refusal.resets_at,
            hard_cap=True,
        )

    def wait_for(self, recording: asyncio.Task) -> None:
        """Hold the charge for cost a task records after the stream ends."""
        self._recordings.add(recording)

    async def record(self, cost_microdollars: int) -> None:
        self.spent_microdollars += cost_microdollars
        if self._ended:
            # Recorded after the turn was charged: charged as it lands.
            await self._charge_due()

    async def end(self) -> None:
        """Charge what the turn recorded, once its late recordings are in."""
        if self._recordings:
            await asyncio.wait(self._recordings, timeout=_COST_SETTLE_TIMEOUT_SECONDS)
        self._ended = True
        await self._charge_due()

    async def _charge_due(self) -> None:
        async with self._charging:
            owed = min(credits_for(self.spent_microdollars), self.cap_credits)
            due = owed - self.charged_credits
            if due <= 0:
                return
            self.charged_credits += await self._spend(due)

    async def _spend(self, credits: int) -> int:
        """Debit up to ``credits``, never below zero; returns what was debited."""
        metadata = UsageTransactionMetadata(
            chat_session_id=self.session_id,
            block_id=AUTOPILOT_BLOCK_ID,
            block="AutoPilotBlock",
            reason="AutoPilot turn billed to credits: plan allowance used up",
        )
        db = credit_db()
        try:
            await db.spend_credits(
                user_id=self.user_id, cost=credits, metadata=metadata
            )
            return credits
        except InsufficientBalanceError:
            # Other runs spent the balance while this turn ran: it takes what
            # is left rather than overdrawing the account.
            try:
                left = min(credits, await db.get_credits(self.user_id))
                if left > 0:
                    await db.spend_credits(
                        user_id=self.user_id, cost=left, metadata=metadata
                    )
            except Exception:
                left = 0
            logger.warning(
                f"Credit-billed turn in session {self.session_id[:12]} owed "
                f"{credits} credits; the balance covered {max(left, 0)}"
            )
            return max(left, 0)
        except Exception:
            logger.error(
                f"Billing leak: could not charge {credits} credits for the "
                f"credit-billed turn in session {self.session_id[:12]}",
                exc_info=True,
            )
            return 0


async def bill_to_credits(
    *,
    user_id: str | None,
    session_id: str,
    tool_name: str | None,
    refusal: TurnPause,
) -> CreditBilledTurn | None:
    """The credit-billed turn to run instead of ``refusal``, or None to refuse.

    Only the AutoPilot block's turns, and only when the balance covers the
    per-turn spend floor. Raises ``RateLimitUnavailable`` when the balance
    can't be read: like an unreadable allowance, that fails the turn closed.
    """
    if user_id is None or tool_name != AUTOPILOT_BLOCK_TOOL_NAME:
        return None
    try:
        balance = await credit_db().get_credits(user_id)
    except Exception as exc:
        raise RateLimitUnavailable("Credit balance unavailable") from exc
    if balance < math.ceil(PER_TURN_SPEND_FLOOR_USD * CREDITS_PER_USD):
        return None
    ceiling = int(config.claude_agent_max_budget_usd * CREDITS_PER_USD)
    return CreditBilledTurn(
        user_id=user_id,
        session_id=session_id,
        cap_credits=min(balance, ceiling),
        refusal=refusal,
    )


@asynccontextmanager
async def credit_billing(turn: CreditBilledTurn) -> AsyncIterator[None]:
    """Run what's inside billed to credits, and charge it when it ends."""
    token = _credit_billed_turn.set(turn)
    try:
        yield
    finally:
        _credit_billed_turn.reset(token)
        await turn.end()


def credit_billed_turn(user_id: str | None) -> CreditBilledTurn | None:
    """The credit-billed turn the running work belongs to, if any."""
    turn = _credit_billed_turn.get()
    if turn is None or turn.user_id != user_id:
        return None
    return turn


async def record_credit_billed_cost(user_id: str, cost_microdollars: int) -> bool:
    """Put a cost on its credit-billed turn; False when there is none."""
    turn = credit_billed_turn(user_id)
    if turn is None:
        return False
    await turn.record(cost_microdollars)
    return True


def wait_for_cost(recording: asyncio.Task) -> None:
    """Have a credit-billed turn's charge wait for a cost recorded later."""
    turn = _credit_billed_turn.get()
    if turn is not None:
        turn.wait_for(recording)


def credits_for(microdollars: int) -> int:
    """Whole credits, rounded up: part of a cent is still charged."""
    return -(-microdollars // MICRODOLLARS_PER_CREDIT)
