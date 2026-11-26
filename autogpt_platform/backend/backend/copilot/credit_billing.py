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

- holds a dollar of the balance to start, and another each time its spend
  nears what it holds, up to what the balance covered when it started within
  the per-turn ceiling (``claude_agent_max_budget_usd``). Turns running at once
  on one balance never hold more than it has together (``credit_turn_holds``),
  and a turn that can't hold more lands as a turn out of budget does. The CLI
  backstop sits on the most it may hold;
- runs on the SDK engine, the one that enforces a budget;
- records its cost here rather than in the plan's daily and weekly windows;
- is charged as its cost lands: each whole credit as soon as the cost it
  recorded reaches it, and the rest, rounded up to a whole credit, when the
  turn ends. In all it pays the cost it recorded, rounded up, never past what
  it holds, and a process that dies mid-turn leaves under a credit of what it
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
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from backend.copilot.active_turns import MAX_TURN_LIFETIME_SECONDS
from backend.copilot.config import ChatConfig
from backend.copilot.constants import (
    AUTOPILOT_BLOCK_ID,
    AUTOPILOT_BLOCK_TOOL_NAME,
    PER_TURN_SPEND_FLOOR_USD,
)
from backend.copilot.credit_turn_holds import LedgerHolds
from backend.copilot.deferred_cancel import DeferredCancel
from backend.copilot.limit_tracking import turn_surface
from backend.copilot.rate_limit import RateLimitUnavailable, platform_spend_refusal
from backend.copilot.tree import MICRODOLLARS_PER_CREDIT
from backend.copilot.turn_budget import BudgetWindow, TurnBudget
from backend.copilot.turn_pause import PauseReason, TurnPause
from backend.data.credit import UsageTransactionMetadata
from backend.data.db_accessors import credit_db
from backend.data.platform_cost import MICRODOLLARS_PER_USD
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
# What a credit-billed turn holds to start, and adds each time it runs short.
_HOLD_STEP_CREDITS = CREDITS_PER_USD
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
        holds: LedgerHolds | None = None,
        turn_id: str = "",
        held_credits: int | None = None,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        # In an organization's run the organization pays, as for the graph's
        # other blocks.
        self.organization_id = organization_id
        self.team_id = team_id
        # What the balance covered when the turn started, within the ceiling:
        # the most it may hold.
        self.cap_credits = cap_credits
        # What it holds of the balance so far; all of the cap without a ledger.
        self.held_credits = cap_credits if held_credits is None else held_credits
        # The refusal this turn runs instead of: the limit that's used up.
        self.refusal = refusal
        self.spent_microdollars = 0
        self.charged_credits = 0
        self._ended = False
        self._recordings: set[asyncio.Task] = set()
        self._charging = asyncio.Lock()
        # Its ledger's holds (``credit_turn_holds``). Top-ups and lowerings run
        # under ``_charging``, so a lowering never undoes a top-up.
        self._holds = holds
        self._turn_id = turn_id
        self._released = False

    @property
    def budget(self) -> TurnBudget:
        """A hard cap at what the turn holds, which grows (``grow``) up to what
        the balance covered: the turn mustn't overdraw."""
        return TurnBudget(
            usd=self.held_credits / CREDITS_PER_USD,
            grows_to_usd=self.cap_credits / CREDITS_PER_USD,
            limiting=_BUDGET_WINDOW[self.refusal.reason],
            resets_at=self.refusal.resets_at,
            hard_cap=True,
            tier=self.refusal.reading.tier if self.refusal.reading else None,
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
            await self.release()

    async def release(self) -> None:
        """Give back what the turn still holds of the balance; it also lapses
        on its own, so a failure here only delays other turns."""
        if self._holds is None or self._released:
            return
        # Not behind ``_charging``, which a hung debit can hold past the turn's
        # give-up: a write still in flight deletes itself (``_released_meanwhile``).
        self._released = True
        try:
            await self._holds.release(self._turn_id)
        except Exception:
            logger.warning(
                f"Credit hold of session {self.session_id[:12]} not released",
                exc_info=True,
            )

    async def grow(self, model_usd: float) -> float:
        """Hold enough for the model to spend ``model_usd`` beside the cost
        already recorded, and a step more, within the cap; returns what the
        turn holds for the model now, under ``model_usd`` when it can't hold
        that much."""
        async with self._charging:
            need = self.spent_microdollars + round(model_usd * MICRODOLLARS_PER_USD)
            await self._hold_for(credits_for(need), step=_HOLD_STEP_CREDITS)
            return self.model_room_usd()

    def model_room_usd(self) -> float:
        """What the turn holds for the model beside the cost it recorded."""
        held = self.held_credits * MICRODOLLARS_PER_CREDIT
        return (held - self.spent_microdollars) / MICRODOLLARS_PER_USD

    async def can_spend(self) -> bool:
        """Whether the cost the turn recorded is still under what it holds,
        holding a step more once it isn't."""
        async with self._charging:
            whole = self.spent_microdollars // MICRODOLLARS_PER_CREDIT
            await self._hold_for(whole + 1, step=_HOLD_STEP_CREDITS)
            held = self.held_credits * MICRODOLLARS_PER_CREDIT
            return self.spent_microdollars < held

    async def _charge_due(self) -> None:
        async with self._charging:
            owed = self._owed_credits(self._ended)
            await self._hold_for(owed, step=0)
            due = min(owed, self.held_credits) - self.charged_credits
            if due <= 0:
                return
            self.charged_credits += await self._spend(due)
            await self._lower_hold()

    async def _hold_for(self, credits: int, *, step: int) -> None:
        """Hold at least ``credits``, and ``step`` more, within the cap: less
        when the balance has no more free. Runs under ``_charging``."""
        want = min(credits + step, self.cap_credits) - self.held_credits
        if credits <= self.held_credits or want <= 0:
            return
        if self._holds is None or self._released:
            return
        try:
            added = await self._holds.hold(
                self._turn_id,
                read_balance=lambda: _balance(self.user_id, self.organization_id),
                credits=want,
            )
            if not await self._released_meanwhile():
                self.held_credits += added
        except Exception:
            # Holding nothing more lands the turn: it fails closed.
            logger.warning(
                f"Credit hold of session {self.session_id[:12]} not raised",
                exc_info=True,
            )

    async def _lower_hold(self) -> None:
        # Debited credits have left the balance; holding them too would
        # count them twice against the turns that start next.
        if self._holds is None or self._released:
            return
        try:
            await self._holds.lower(
                self._turn_id, self.held_credits - self.charged_credits
            )
            await self._released_meanwhile()
        except Exception:
            logger.warning(
                f"Credit hold of session {self.session_id[:12]} not lowered",
                exc_info=True,
            )

    async def _released_meanwhile(self) -> bool:
        """Whether the turn was released while its last ledger write was in
        flight; that write then rewrote its field, so it is deleted again."""
        if not self._released or self._holds is None:
            return False
        await self._holds.release(self._turn_id)
        return True

    def _owed_credits(self, final: bool) -> int:
        """Whole credits while the turn runs, the part credit too once it has
        ended."""
        spent = self.spent_microdollars
        return credits_for(spent) if final else spent // MICRODOLLARS_PER_CREDIT

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
    run. None where credits are off, leaving no balance to bill, in another
    user's personal organization, or where the SDK engine that caps a turn's
    spend can't run. Raises ``RateLimitUnavailable`` when the balance can't be
    read: like an unreadable allowance, that fails the turn closed.
    """
    if user_id is None or tool_name != AUTOPILOT_BLOCK_TOOL_NAME:
        return None
    if not credits_can_pay_block_turns():
        return None
    try:
        # The turn's ledger is decided once, here.
        if organization_id:
            owner = await credit_db().get_personal_org_owner(organization_id)
            if owner == user_id:
                organization_id = team_id = None
            elif owner is not None:
                return None  # another user's personal org bills nobody
        ceiling = int(config.claude_agent_max_budget_usd * CREDITS_PER_USD)
        balances: list[int] = []

        async def read_balance() -> int:
            balances.append(await _balance(user_id, organization_id))
            return balances[-1]

        # A turn holds part of what the turns already running on this ledger
        # don't, so turns started together can't each be offered it all.
        holds = LedgerHolds(organization_id or user_id, MAX_TURN_LIFETIME_SECONDS)
        turn_id = str(uuid.uuid4())
        held = await holds.hold(
            turn_id,
            read_balance=read_balance,
            credits=min(_HOLD_STEP_CREDITS, ceiling),
            at_least=math.ceil(PER_TURN_SPEND_FLOOR_USD * CREDITS_PER_USD),
        )
    except Exception as exc:
        raise RateLimitUnavailable("Credit balance unavailable") from exc
    if not held:
        return None
    return CreditBilledTurn(
        user_id=user_id,
        session_id=session_id,
        # What the balance covered as the hold was taken, within the ceiling.
        cap_credits=min(balances[-1], ceiling),
        refusal=refusal,
        organization_id=organization_id,
        team_id=team_id,
        holds=holds,
        turn_id=turn_id,
        held_credits=held,
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
        # On a task of its own: a cancel that lands while the turn is charged
        # (a Stop's second one, #17286) waits for the charge, then goes on,
        # past an error from the charge too, which it keeps as its context.
        deferred = DeferredCancel()
        try:
            await deferred.finish(turn.end())
        finally:
            if deferred.held is not None:
                raise deferred.held


async def tool_spend_refusal(user_id: str | None) -> str | None:
    """Why a tool may not pay a provider on the platform's key now; None when it may.

    In a credit-billed turn the tool's cost lands on the turn's credits, so it
    may spend until the turn's recorded cost reaches what it can hold.
    """
    turn = credit_billed_turn(user_id)
    if turn is None:
        return await platform_spend_refusal(user_id, surface=turn_surface())
    if await turn.can_spend():
        return None
    return "This run has used all the credits it can spend."


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
