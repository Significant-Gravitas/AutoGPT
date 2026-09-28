"""The shapes of a ``DreamPass`` row: what a new pass inserts, what one
transition writes, and the row as it reads back.

The JSON columns hold the dream's own models; the input bundle keeps the
format the batch path persists to Redis (``copilot/dream/input_bundle.py``).
How an update lands on a row, only ever moving it forward, is
``dream_pass_update.py``.
"""

from datetime import datetime
from typing import Literal

import prisma.models
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import BaseModel, Field, model_validator

from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.input_bundle import input_bundle_from_dict
from backend.copilot.dream.schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassUsage,
    IngestionDrainStatus,
    RecombinationOutput,
)

# The statuses a row can still move on from. Any other status is terminal: the
# row is final, and only the cleanup after its pass finishing writes it again
# (``DreamPassUpdate.closed_row``).
OPEN_STATUSES: tuple[DreamPassStatus, ...] = (
    DreamPassStatus.QUEUED,
    DreamPassStatus.RUNNING,
    DreamPassStatus.SUBMITTED,
    DreamPassStatus.APPLYING,
)
CLOSED_STATUSES: tuple[DreamPassStatus, ...] = tuple(
    status for status in DreamPassStatus if status not in OPEN_STATUSES
)

# A new row's cancelGeneration (the column default). Only a stop from outside
# the pass moves it, so a pass that reads anything else was stopped.
INITIAL_CANCEL_GENERATION = 0

# The nullable columns an update may empty, by their field names.
ClearableColumn = Literal[
    "lease_token", "lease_expires_at", "input_bundle", "cleanup_pending_at"
]

# What a row drops as it closes: a finished pass holds no lease, and nothing
# resumes it from its input bundle, the bulk of an open batch row. Its phase
# outputs, operations and usage stay.
CLOSED_ROW_CLEARS: frozenset[ClearableColumn] = frozenset(
    {"lease_token", "lease_expires_at", "input_bundle"}
)
# What a row marked for a cleanup drops as it closes: the bundle only. It
# keeps its lease until the cleanup has finished: the token to release the
# pass's lock with, and the expiry that says whether the pass may still run.
MARKED_ROW_CLEARS: frozenset[ClearableColumn] = frozenset({"input_bundle"})


class DreamPhaseOutputs(BaseModel):
    """Each phase's validated output, as the next phase and apply read it."""

    consolidate: ConsolidationOutput | None = None
    recombine: RecombinationOutput | None = None
    sanitize: DreamOperations | None = None


class DreamPassApplied(BaseModel):
    """What apply reported: the counts and narrative a ``DreamPassResult``
    carries, and the per-operation snapshot."""

    consolidated_count: int = 0
    proposal_count: int = 0
    demotion_count: int = 0
    entity_invalidation_count: int = 0
    # Demotions the recall guard dropped; a row written before it reads 0.
    protected_demotions: int = 0
    summary_for_user: str = ""
    dream_session_id: str | None = None
    ingestion_drain_status: IngestionDrainStatus = IngestionDrainStatus.drained
    snapshot: DreamOperationsSnapshot | None = None


class DreamPassOperations(BaseModel):
    """The clamped operations apply was handed, then what it reported."""

    planned: DreamOperations | None = None
    applied: DreamPassApplied | None = None


class DreamPassDraft(BaseModel):
    """A new row: whose pass it is, how it runs, what started it, and the
    lease it starts with (the token of the lock it is about to take)."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    expert_id: str | None = None
    scope_key: str = Field(min_length=1)
    route: DreamPassRoute
    trigger: DreamPassTrigger
    status: DreamPassStatus = DreamPassStatus.RUNNING
    phase: DreamPassPhase = DreamPassPhase.GATHER
    started_at: datetime | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None


class DreamPassUpdate(BaseModel):
    """What one transition writes. A ``None`` field leaves its column as it is.

    ``status`` and ``phase`` only ever move forward. ``provider_batch_id`` is
    the batch of the update's ``phase``, which it therefore needs.
    ``phase_outputs`` and ``operations`` merge into what the row holds, one
    top-level field at a time, so a transition sends only the phase or the
    part it produced.

    A stop from outside the pass (a cancel, an expiry) sets
    ``bump_cancel_generation``, and may make the write conditional on more
    than the row being open: ``owner_user_id`` (the row is that user's) and
    ``not_updated_since`` (nothing has written the row after that instant).

    ``clear`` names the nullable columns the update empties, which a ``None``
    field cannot say; every transition that closes a row clears
    ``CLOSED_ROW_CLEARS``, or ``MARKED_ROW_CLEARS`` when it marks the row for
    a cleanup (``cleanup_pending_at``). A column is given a value or cleared,
    not both.

    ``closed_row`` is the one write a closed row takes: the cleanup after its
    pass saying it has finished. It applies to a closed row only, and may
    only clear columns.
    """

    status: DreamPassStatus | None = None
    phase: DreamPassPhase | None = None
    skip_reason: str | None = None
    provider_batch_id: str | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    input_bundle: DreamInput | None = None
    phase_outputs: DreamPhaseOutputs | None = None
    operations: DreamPassOperations | None = None
    usage: DreamPassUsage | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    error: str | None = None
    submitted_at: datetime | None = None
    applied_at: datetime | None = None
    completed_at: datetime | None = None
    bump_cancel_generation: bool = False
    owner_user_id: str | None = None
    not_updated_since: datetime | None = None
    cleanup_pending_at: datetime | None = None
    clear: frozenset[ClearableColumn] = frozenset()
    closed_row: bool = False

    @model_validator(mode="after")
    def _batch_names_its_phase(self) -> "DreamPassUpdate":
        if self.provider_batch_id is not None and self.phase is None:
            raise ValueError("a provider batch id needs the phase it runs")
        return self

    @model_validator(mode="after")
    def _set_or_cleared(self) -> "DreamPassUpdate":
        given: dict[ClearableColumn, bool] = {
            "lease_token": self.lease_token is not None,
            "lease_expires_at": self.lease_expires_at is not None,
            "input_bundle": self.input_bundle is not None,
            "cleanup_pending_at": self.cleanup_pending_at is not None,
        }
        both = sorted(column for column in self.clear if given[column])
        if both:
            raise ValueError(f"set and cleared at once: {', '.join(both)}")
        return self

    @model_validator(mode="after")
    def _closed_row_only_clears(self) -> "DreamPassUpdate":
        if not self.closed_row:
            return self
        written = self.model_dump(
            exclude={"clear", "closed_row"}, exclude_defaults=True
        )
        if written:
            raise ValueError(
                f"a closed row only has columns cleared: {sorted(written)}"
            )
        return self


class DreamPassRecord(BaseModel):
    """One ``DreamPass`` row."""

    id: str
    user_id: str
    expert_id: str | None
    scope_key: str
    route: DreamPassRoute
    trigger: DreamPassTrigger
    phase: DreamPassPhase
    status: DreamPassStatus
    skip_reason: str | None
    cancel_generation: int
    provider_batch_id: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    cleanup_pending_at: datetime | None = None
    input_bundle: DreamInput | None
    phase_outputs: DreamPhaseOutputs
    operations: DreamPassOperations
    usage: DreamPassUsage | None
    window_start: datetime | None
    window_end: datetime | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    submitted_at: datetime | None
    applied_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.DreamPass) -> "DreamPassRecord":
        return cls(
            id=row.id,
            user_id=row.userId,
            expert_id=row.expertId,
            scope_key=row.scopeKey,
            route=row.route,
            trigger=row.trigger,
            phase=row.phase,
            status=row.status,
            skip_reason=row.skipReason,
            cancel_generation=row.cancelGeneration,
            provider_batch_id=row.providerBatchId,
            lease_token=row.leaseToken,
            lease_expires_at=row.leaseExpiresAt,
            cleanup_pending_at=row.cleanupPendingAt,
            input_bundle=(
                input_bundle_from_dict(dict(row.inputBundle))
                if row.inputBundle is not None
                else None
            ),
            phase_outputs=DreamPhaseOutputs.model_validate(row.phaseOutputs or {}),
            operations=DreamPassOperations.model_validate(row.operations or {}),
            usage=(
                DreamPassUsage.model_validate(row.usage)
                if row.usage is not None
                else None
            ),
            window_start=row.windowStart,
            window_end=row.windowEnd,
            error=row.error,
            created_at=row.createdAt,
            started_at=row.startedAt,
            submitted_at=row.submittedAt,
            applied_at=row.appliedAt,
            completed_at=row.completedAt,
            updated_at=row.updatedAt,
        )
