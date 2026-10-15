"""One dream pass while the sync entry point runs it: who and what it is,
when it started, and what its phases have used so far.

Built once when a pass starts and handed down, so every result the pass can
end with (a skip, a failure at any step, an unexpected error, the hand-off to
the batch route) carries the same identity and timing, and every failure
carries the usage of every phase billed before it.
"""

import asyncio
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .routing import ExecutionPath
from .schemas import DreamPassResult, DreamPassUsage, PhaseUsage
from .usage import aggregate_usage


class DreamPassRun(BaseModel):
    user_id: str
    pass_id: str
    started_at: datetime
    monotonic_start: float
    execution_path: ExecutionPath
    # Every phase billed so far, in order, a failed phase whose answer came
    # back included.
    phases: list[PhaseUsage] = Field(default_factory=list)

    @classmethod
    def begin(cls, user_id: str, execution_path: ExecutionPath) -> "DreamPassRun":
        return cls(
            user_id=user_id,
            pass_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
            monotonic_start=asyncio.get_event_loop().time(),
            execution_path=execution_path,
        )

    def usage(self) -> DreamPassUsage:
        """What the phases billed so far used, per phase and in total."""
        return aggregate_usage(self.phases, self.execution_path)

    def elapsed_seconds(self) -> float:
        return asyncio.get_event_loop().time() - self.monotonic_start

    def failure(self, error: str) -> DreamPassResult:
        """The pass failed. On the sync route it still carries the usage of
        the phases billed before the error: billing charges for tokens we
        already paid for. The batch route's phases bill at the provider and
        reach the callbacks, never this side (a batch submitted before the
        failure still runs), so its usage here is unknown."""
        if self.execution_path == "anthropic_batch":
            return self._ended(error=error)
        return self._ended(error=error, usage=self.usage())

    def skipped(self, reason: str) -> DreamPassResult:
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
