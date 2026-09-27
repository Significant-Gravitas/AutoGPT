"""One dream pass while the sync entry point runs it: who and what it is,
when it started, the lease it holds, and what its phases have used so far.

Built once when a pass starts and handed down, so every result the pass can
end with (a skip, a failure at any step, a stop from outside, an unexpected
error, the hand-off to the batch route) carries the same identity and timing,
and every failure carries the usage of every phase billed before it. A step
that ends the pass early raises ``PassEnded`` with the result it ends with.

The pass mints its lease token with its id, so its row carries the lease from
the first write; the scope's lock is then taken under that token and held by
the run (``hold``) for the renewals at each step (``lease.py``).
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from backend.copilot.inference.context import InferenceError

from .locks import DreamLockHandle
from .routing import ExecutionPath
from .schemas import DreamPassResult, DreamPassUsage, DreamPhase, PhaseUsage
from .usage import aggregate_usage, phase_usage

# Why a pass was skipped rather than run: its scope was taken (the lock held,
# or another of its passes still open and fresh), the master flag was off, the
# budget said no, or there was nothing (new) to dream about.
DreamSkipReason = Literal[
    "lock_held",
    "pass_in_progress",
    "disabled",
    "insufficient_credits",
    "no_input",
    "no_new_activity",
]


class DreamPassRun(BaseModel):
    user_id: str
    pass_id: str
    started_at: datetime
    monotonic_start: float
    execution_path: ExecutionPath
    # An admin asked the guard to expire a fresh open pass of the scope
    # rather than skip behind it.
    force: bool = False
    # Every phase billed so far, in order, a failed phase whose answer came
    # back included.
    phases: list[PhaseUsage] = Field(default_factory=list)
    # The ownership token of the scope's dream lock the pass takes.
    lease_token: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _lock: DreamLockHandle | None = PrivateAttr(default=None)

    @classmethod
    def begin(
        cls, user_id: str, execution_path: ExecutionPath, *, force: bool = False
    ) -> "DreamPassRun":
        return cls(
            user_id=user_id,
            pass_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
            monotonic_start=asyncio.get_event_loop().time(),
            execution_path=execution_path,
            force=force,
        )

    @property
    def lock(self) -> DreamLockHandle | None:
        """The scope's lock the pass holds, once it has taken it."""
        return self._lock

    def hold(self, lock: DreamLockHandle) -> None:
        self._lock = lock

    def usage(self) -> DreamPassUsage:
        """What the phases billed so far used, per phase and in total."""
        return aggregate_usage(self.phases, self.execution_path)

    def bill_failed_phase(self, phase: DreamPhase, exc: InferenceError) -> None:
        """Count a failed phase among the billed ones when its answer came
        back (the phase recorded its cost); a call that got none used
        nothing."""
        if exc.usage is not None:
            self.phases.append(phase_usage(phase, exc.usage))

    def elapsed_seconds(self) -> float:
        return asyncio.get_event_loop().time() - self.monotonic_start

    def failure(self, error: str) -> DreamPassResult:
        """The pass failed, or was stopped from outside. On the sync route it
        still carries the usage of the phases billed before the error: billing
        charges for tokens we already paid for. The batch route's phases bill
        at the provider and reach the callbacks, never this side (a batch
        submitted before the failure still runs), so its usage here is
        unknown."""
        if self.execution_path == "anthropic_batch":
            return self._ended(error=error)
        return self._ended(error=error, usage=self.usage())

    def skipped(self, reason: DreamSkipReason) -> DreamPassResult:
        return self._ended(skipped=True, skip_reason=reason)

    def handed_off(self) -> DreamPassResult:
        """Phase 1 is queued on the batch route: no usage, operations or
        session yet. The batch callbacks deliver those when phase 3 lands and
        apply runs."""
        return self._ended()

    def _ended(
        self,
        *,
        error: str | None = None,
        skipped: bool = False,
        skip_reason: str | None = None,
        usage: DreamPassUsage | None = None,
    ) -> DreamPassResult:
        return DreamPassResult(
            user_id=self.user_id,
            pass_id=self.pass_id,
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc),
            elapsed_seconds=self.elapsed_seconds(),
            execution_path=self.execution_path,
            error=error,
            skipped=skipped,
            skip_reason=skip_reason,
            usage=usage,
        )


class PassEnded(Exception):
    """A pass that ends before apply, carrying the result it ends with."""

    def __init__(self, result: DreamPassResult) -> None:
        super().__init__(result.error or result.skip_reason)
        self.result = result
