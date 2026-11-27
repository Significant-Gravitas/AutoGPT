"""Decides when a running turn has to land, and records why it paused.

Two limits now end a turn on purpose instead of by accident: the money it may
spend (``TurnBudget``) and the number of model calls before a check-in. Close
to either, the governor switches to *landing*: every further tool call is
refused with a reason that asks for a checkpoint, so the agent stops at a
clean point and says where it got to. A warning alone doesn't do that --
models carried on past one in every test -- but a refused tool does, and the
refusal reaches the model inside the tool result on every transport.

The CLI's own limits stay, sized as backstops that fire only if landing
didn't: ``max_budget_usd`` above the budget (in the CLI's price units) and
``max_turns`` a little past the check-in.
"""

from collections.abc import Awaitable, Callable

from claude_agent_sdk import AssistantMessage, Message, StreamEvent

from backend.copilot.constants import PER_TURN_SPEND_FLOOR_USD
from backend.copilot.limit_tracking import LimitReading
from backend.copilot.turn_budget import TurnBudget
from backend.copilot.turn_pause import PauseReason, TurnPause

from .usage_meter import UsageMeter, cli_price_ratio

# Model calls past the check-in before the CLI's own max_turns stops the turn.
STEP_BACKSTOP_MARGIN = 50
# Spend held back for the checkpoint: about two calls at the latest call's
# price (the round whose tools are refused, then the reply), within bounds.
_RESERVE_MIN_USD = 0.02
_RESERVE_MAX_SHARE = 0.25
# Backstop headroom past the budget, paid plans only: a trial's allowance is
# a hard cap, so its backstop sits exactly on it.
_BACKSTOP_MIN_USD = 0.25
_BACKSTOP_SHARE = 0.15

_BUDGET_REFUSAL = (
    "Tool calls are paused: this task has used the budget it has for now, so "
    "every further tool call will be refused. Don't call any more tools. Reply "
    "to the user now with a short checkpoint in this form:\n"
    "Where I got to: <what is done>\n"
    "What's left: <what remains>\n"
    "Next step: <the exact next action>\n"
    "The user can pick the task up from there. Don't quote costs or budget "
    "figures."
)

_STEPS_REFUSAL = (
    "Tool calls are paused for a check-in: this task has run for {steps} "
    "steps, so every further tool call will be refused. Don't call any more "
    "tools. Reply to the user now with a short update in this form:\n"
    "Progress: <what is done so far>\n"
    "Next step: <the exact next action>\n"
    "The user can tell you to keep going from there."
)


class TurnGovernor:
    """Per-turn spend and step tracking, and the landing decision.

    Lives for the whole turn, across retries: money spent on an attempt that
    failed was still spent.
    """

    def __init__(
        self,
        *,
        budget: TurnBudget | None,
        step_checkin: int,
        grow: Callable[[float], Awaitable[float]] | None = None,
        room: Callable[[], float] | None = None,
    ) -> None:
        # None on routes the account doesn't pay for (a ChatGPT subscription):
        # those only get the step check-in.
        self.budget = budget
        # For a budget that grows (``CreditBilledTurn.grow``): asked to cover
        # the spend plus the reserve once spend nears the budget, it answers
        # the budget it could reach.
        self.grow = grow
        # What such a budget holds for the model now: cost recorded beside the
        # model's, a tool's, shrinks it between two ``grow`` calls.
        self.room = room
        self.step_checkin = step_checkin
        self.meter = UsageMeter()
        self.landing: PauseReason | None = None
        # Set once a tool call was refused: from then on the turn is landing
        # for real. A threshold crossed by the final answer, with no tool
        # call after it, finished the task rather than pausing it.
        self.refused = False
        # Set once the turn has closed as a pause: queued follow-ups then wait
        # for the user to pick the task back up instead of auto-continuing.
        self.paused = False
        # Set once a budget that grows was spent past (``stop``).
        self.stopped = False

    def observe(self, message: Message) -> None:
        if isinstance(message, AssistantMessage):
            self.meter.observe(message)
        elif isinstance(message, StreamEvent):
            self.meter.observe_stream(message)
        else:
            return
        self._update()

    async def keep_up(self) -> None:
        """Grow a budget that can grow once spend nears it, and land the turn
        when it can't grow past the reserve."""
        budget = self._live_budget()
        if self.grow is None or budget is None or self.landing is not None:
            return
        if not self._near(budget):
            return
        usd = await self.grow(self.meter.spent_usd + self._reserve_usd(budget))
        self.budget = budget = budget.model_copy(update={"usd": usd})
        if self.landing is None and self._near(budget):
            self.landing = budget.limiting

    def stop(self) -> bool:
        """True, once, when a budget that grows has landed and its spend has
        passed it. The CLI's backstop sits on the most such a budget can grow
        to, so the caller interrupts the CLI here instead."""
        budget = self._live_budget()
        if self.stopped or self.grow is None or budget is None:
            return False
        if self.landing is None or self.meter.spent_usd < budget.usd:
            return False
        self.stopped = True
        return True

    def refusal(self) -> str | None:
        """Why the next tool call is refused, or None to let it run."""
        self._update()
        if self.landing is None:
            return None
        self.refused = True
        if self.landing == "steps":
            return _STEPS_REFUSAL.format(steps=self.meter.main_calls)
        return _BUDGET_REFUSAL

    def pause(self, result_subtype: str, *, wrote_checkpoint: bool) -> TurnPause | None:
        """The pause this turn ended in, or None if it finished or failed.

        A backstop subtype means the CLI stopped the turn itself, before (or
        while) the agent wrote its checkpoint, so none is claimed; the same for
        a turn stopped at a budget that grows (``stop``), whatever its CLI
        reported. A genuine error while landing stays an error, and a turn
        that ended normally without a refused tool simply finished: only a
        turn that ended normally after its tools were refused is a pause.
        """
        if self.stopped or result_subtype == "error_max_budget_usd":
            return self._pause(self._spend_reason(), checkpoint_saved=False)
        if result_subtype == "error_max_turns":
            return self._pause("steps", checkpoint_saved=False)
        if self.landing is None or not self.refused:
            return None
        if result_subtype != "success":
            return None
        return self._pause(self.landing, checkpoint_saved=wrote_checkpoint)

    def hangup_pause(self) -> TurnPause | None:
        """The pause for a turn the CLI ended without a result while landing.

        A hangup is otherwise a crash and keeps its own notice; once the tools
        were being refused it is the landing that ended, so it still pauses.
        """
        if self.stopped:
            return self._pause(self._spend_reason(), checkpoint_saved=False)
        if self.landing is None or not self.refused:
            return None
        return self._pause(self.landing, checkpoint_saved=False)

    @property
    def landed(self) -> bool:
        """True once a tool was refused, whether or not it closed as a pause."""
        return self.paused or self.refused

    @property
    def cli_max_turns(self) -> int:
        return self.step_checkin + STEP_BACKSTOP_MARGIN

    def cli_budget_usd(self, model: str | None) -> float | None:
        """The CLI's ``max_budget_usd``: the budget plus headroom, in CLI units."""
        if self.budget is None:
            return None
        if self.budget.hard_cap:
            most = self.budget.grows_to_usd or self.budget.usd
            return most * cli_price_ratio(model)
        headroom = max(_BACKSTOP_MIN_USD, _BACKSTOP_SHARE * self.budget.usd)
        cli_usd = (self.budget.usd + headroom) * cli_price_ratio(model)
        return max(PER_TURN_SPEND_FLOOR_USD, cli_usd)

    def _update(self) -> None:
        if self.landing is not None:
            return
        if self.meter.main_calls >= self.step_checkin:
            self.landing = "steps"
        elif self.grow is None and self.budget is not None and self._near(self.budget):
            # A budget that grows lands only once ``keep_up`` can't grow it.
            self.landing = self.budget.limiting

    def _live_budget(self) -> TurnBudget | None:
        if self.room is not None and self.budget is not None:
            usd = self.room()
            if usd != self.budget.usd:
                self.budget = self.budget.model_copy(update={"usd": usd})
        return self.budget

    def _near(self, budget: TurnBudget) -> bool:
        return self.meter.spent_usd >= budget.usd - self._reserve_usd(budget)

    def _spend_reason(self) -> PauseReason:
        return self.budget.limiting if self.budget is not None else "task"

    def _reserve_usd(self, budget: TurnBudget) -> float:
        latest = 2 * self.meter.last_main_call_usd
        return min(_RESERVE_MAX_SHARE * budget.usd, max(_RESERVE_MIN_USD, latest))

    def _pause(self, reason: PauseReason, *, checkpoint_saved: bool) -> TurnPause:
        self.paused = True
        resets_at = None
        if reason in ("daily", "weekly") and self.budget is not None:
            resets_at = self.budget.resets_at
        return TurnPause(
            reason=reason,
            resets_at=resets_at,
            steps=self.meter.main_calls if reason == "steps" else None,
            checkpoint_saved=checkpoint_saved,
            reading=(
                LimitReading(tier=self.budget.tier)
                if self.budget is not None and self.budget.tier
                else None
            ),
        )
