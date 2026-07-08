"""Bounded re-plan escalation for the executor loop.

Between executor loop rounds the baseline path asks a :class:`ReplanController`
whether the plan needs revising — because the executor emitted an explicit
``[[REPLAN]]`` signal or because it hit a run of tool failures.  The controller
owns the per-turn cap so the escalation cost is bounded, and is deliberately
pure of any streaming/session plumbing so it can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from backend.copilot.config import ChatConfig
from backend.copilot.planner.models import Plan
from backend.copilot.planner.service import (
    REPLAN_SIGNAL,
    PlannerUsage,
    plan_to_executor_prompt,
    revise_plan,
)

logger = logging.getLogger(__name__)

# Max re-plans per conversation turn — after this the executor continues
# best-effort in the plain loop.
MAX_REPLANS = 2
# Consecutive tool failures that trigger an automatic re-plan even without an
# explicit signal from the model.
REPLAN_ERROR_THRESHOLD = 3

ReplanAction = Literal["revised", "capped", "revision_failed", "none"]


def _extract_signal_reason(text: str) -> str:
    """Grab the one-line reason the executor wrote after ``REPLAN_SIGNAL``."""
    idx = text.find(REPLAN_SIGNAL)
    if idx == -1:
        return ""
    tail = text[idx + len(REPLAN_SIGNAL) :].strip()
    return tail.splitlines()[0].strip() if tail else ""


def evaluate_replan_trigger(
    new_text: str | None,
    consecutive_tool_errors: int,
) -> str | None:
    """Decide whether to re-plan; returns a human reason string or ``None``.

    Two triggers: an explicit ``REPLAN_SIGNAL`` in the executor's latest text,
    or ``REPLAN_ERROR_THRESHOLD`` consecutive tool failures.
    """
    text = new_text or ""
    if REPLAN_SIGNAL in text:
        return (
            _extract_signal_reason(text) or "executor signalled the plan is unworkable"
        )
    if consecutive_tool_errors >= REPLAN_ERROR_THRESHOLD:
        return f"{consecutive_tool_errors} consecutive tool failures"
    return None


@dataclass
class ReplanOutcome:
    """Result of one :meth:`ReplanController.maybe_revise` call.

    ``action``:
      - ``"none"``    — no trigger; do nothing.
      - ``"revised"`` — append ``system_message`` to the conversation and
        persist ``plan``; the executor resumes on the revised plan.
      - ``"capped"``  — a trigger fired but the per-turn cap is spent; the
        caller should reset the failure streak and continue best-effort.
      - ``"revision_failed"`` — a revision was attempted but the planner
        returned nothing; continue best-effort.

    ``usage`` carries the tokens/cost of any re-plan LLM call made this round
    (empty for ``"none"``/``"capped"``) so the caller can bill it into the
    turn's usage — the same channel the frontend token display reads.
    """

    action: ReplanAction
    reason: str | None = None
    system_message: str | None = None
    plan: Plan | None = None
    usage: PlannerUsage = field(default_factory=PlannerUsage)


class ReplanController:
    """Owns the per-turn re-plan cap, scan cursor, and current plan.

    Stateful across executor rounds within a single turn; a fresh instance is
    created per turn.  When constructed with ``plan=None`` (the split is off or
    the turn wasn't multi-step) :meth:`maybe_revise` is always a no-op.
    """

    def __init__(
        self,
        plan: Plan | None,
        *,
        planner_model: str,
        user_id: str | None,
        session_id: str | None,
        config: ChatConfig,
    ) -> None:
        self.plan = plan
        self.replans_used = 0
        self.capped = False
        self._scan_pos = 0
        self._planner_model = planner_model
        self._user_id = user_id
        self._session_id = session_id
        self._config = config

    async def maybe_revise(
        self,
        *,
        executor_text: str,
        consecutive_tool_errors: int,
        tools: Sequence[Any],
    ) -> ReplanOutcome:
        """Inspect executor progress since the last call and maybe re-plan.

        Advances the internal scan cursor over ``executor_text`` so a signal /
        failure is only acted on once.  Enforces :data:`MAX_REPLANS`.
        """
        if self.plan is None or self.capped:
            return ReplanOutcome("none")

        new_text = executor_text[self._scan_pos :]
        self._scan_pos = len(executor_text)
        reason = evaluate_replan_trigger(new_text, consecutive_tool_errors)
        if reason is None:
            return ReplanOutcome("none")

        if self.replans_used >= MAX_REPLANS:
            self.capped = True
            return ReplanOutcome("capped", reason=reason)

        revised, usage = await revise_plan(
            plan=self.plan,
            failure_reason=reason,
            tools=tools,
            planner_model=self._planner_model,
            user_id=self._user_id,
            config=self._config,
            session_id=self._session_id,
        )
        if revised is None:
            return ReplanOutcome("revision_failed", reason=reason, usage=usage)

        self.plan = revised
        self.replans_used += 1
        return ReplanOutcome(
            "revised",
            reason=reason,
            system_message=plan_to_executor_prompt(revised, revised=True),
            plan=revised,
            usage=usage,
        )
