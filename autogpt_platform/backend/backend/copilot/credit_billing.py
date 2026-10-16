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
- is charged as its cost lands: each whole credit as soon as the cost it
  recorded reaches it, and the rest, rounded up to a whole credit, when the
  turn ends. In all it pays the cost it recorded, rounded up, never past its
  cap, and a process that dies mid-turn leaves under a credit of what it
  recorded unpaid. A turn that ended without a result is charged what it
  recorded too, through the same accounting as any other turn;
- counts a debit that failed for a reason other than the balance as charged,
  and logs it as a possible leak. Such a debit may still have landed: the
  credit model commits it before its auto top-up runs, and a reply can be
  lost, so taking it again could charge it twice. The database client already
  retries what fails before reaching the database. When other runs spent the
  balance meanwhile, the turn takes what is left and never overdraws it.
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
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
config = ChatConfig()
settings = Settings()

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
        organization_id: str | None = None,
        team_id: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        # In an organization's run the organization pays, as for the graph's
        # other blocks.
        self.organization_id = organization_id
        self.team_id = team_id
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
        """Put a cost on the turn and charge what it completes: whole credits
        while the turn runs, the rest rounded up once it has ended. Shielded
        like the end's charge, so a debit already sent is counted even if the
        caller is cancelled, and no later charge takes it again."""
        self.spent_microdollars += cost_microdollars
        await asyncio.shield(self._charge_due())

    async def end(self) -> None:
        """Charge the rest of what the turn recorded, rounded up, once its
        late recordings are in.

        A cancel during the wait doesn't skip the charge: the turn still pays
        for what it spent, and cost recorded later is charged as it lands.
        """
        try:
            if self._recordings:
                await asyncio.wait(
                    self._recordings, timeout=_COST_SETTLE_TIMEOUT_SECONDS
                )
        finally:
            self._ended = True
            await asyncio.shield(self._charge_due())

    async def _charge_due(self) -> None:
        async with self._charging:
            final = self._ended
            due = self._owed_credits(final) - self.charged_credits
            if due <= 0:
                return
            self.charged_credits += await self._spend(due)

    def _owed_credits(self, final: bool) -> int:
        """Whole credits while the turn runs, the part credit too once it has
        ended; never past the cap."""
        spent = self.spent_microdollars
        owed = credits_for(spent) if final else spent // MICRODOLLARS_PER_CREDIT
        return min(owed, self.cap_credits)

    async def _spend(self, credits: int) -> int:
        """Debit up to ``credits``, never below zero; returns what the turn
        counts as charged, which a debit that may have landed is."""
        try:
            await self._debit(credits)
        except InsufficientBalanceError:
            return await self._spend_what_is_left(credits)
        except Exception:
            self._log_unconfirmed(credits)
        return credits

    async def _spend_what_is_left(self, credits: int) -> int:
        # Other runs spent the balance while this turn ran: it takes what is
        # left rather than overdrawing the account. A balance it can't read
        # took nothing, so what is due stays due.
        try:
            left = max(
                min(credits, await _balance(self.user_id, self.organization_id)), 0
            )
        except Exception:
            left = 0
        if left > 0:
            try:
                await self._debit(left)
            except InsufficientBalanceError:
                left = 0
            except Exception:
                self._log_unconfirmed(left)
        logger.warning(
            f"Credit-billed turn in session {self.session_id[:12]} owed "
            f"{credits} credits; the balance covered {left}"
        )
        return left

    def _log_unconfirmed(self, credits: int) -> None:
        logger.error(
            f"Billing leak: could not confirm a charge of {credits} credits for "
            f"the credit-billed turn in session {self.session_id[:12]}; it "
            "counts as charged, since taking it again could charge it twice",
            exc_info=True,
        )

    async def _debit(self, credits: int) -> None:
        metadata = UsageTransactionMetadata(
            chat_session_id=self.session_id,
            block_id=AUTOPILOT_BLOCK_ID,
            block="AutoPilotBlock",
            reason="AutoPilot turn billed to credits: plan allowance used up",
        )
        if self.organization_id:
            await credit_db().spend_org_credits(
                org_id=self.organization_id,
                user_id=self.user_id,
                cost=credits,
                metadata=metadata,
                team_id=self.team_id,
            )
        else:
            await credit_db().spend_credits(
                user_id=self.user_id, cost=credits, metadata=metadata
            )


async def bill_to_credits(
    *,
    user_id: str | None,
    session_id: str,
    tool_name: str | None,
    refusal: TurnPause,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> CreditBilledTurn | None:
    """The credit-billed turn to run instead of ``refusal``, or None to refuse.

    Only the AutoPilot block's turns, and only when the balance covers the
    per-turn spend floor: the organization's balance in an organization's
    run. None where credits are off, leaving no balance to bill, or where
    the SDK engine that caps a turn's spend can't run. Raises
    ``RateLimitUnavailable`` when the balance can't be read: like an
    unreadable allowance, that fails the turn closed.
    """
    if user_id is None or tool_name != AUTOPILOT_BLOCK_TOOL_NAME:
        return None
    if not credits_can_pay_block_turns():
        return None
    try:
        balance = await _balance(user_id, organization_id)
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
        organization_id=organization_id,
        team_id=team_id,
    )


def credits_can_pay_block_turns() -> bool:
    """Whether a block turn past the allowance can run on credits here: credits
    are on, and the SDK engine that caps a turn's spend can run."""
    return settings.config.enable_credit and config.thinking_available


async def _balance(user_id: str, organization_id: str | None) -> int:
    if organization_id:
        return await credit_db().get_org_credits(org_id=organization_id)
    return await credit_db().get_credits(user_id)


@asynccontextmanager
async def credit_billing(turn: CreditBilledTurn) -> AsyncIterator[None]:
    """Run what's inside billed to credits, and charge the rest when it ends."""
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
