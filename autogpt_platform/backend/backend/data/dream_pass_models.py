"""The shapes of a ``DreamPass`` row: what a new pass inserts, what one
transition writes, and the row as it reads back.

The JSON columns hold the dream's own models; the input bundle keeps the
format the batch path persists to Redis (``copilot/dream/input_bundle.py``).
How an update lands on a row, only ever moving it forward, is
``dream_pass_update.py``.
"""

from datetime import datetime

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
# row is final and no update writes it.
OPEN_STATUSES: tuple[DreamPassStatus, ...] = (
    DreamPassStatus.QUEUED,
    DreamPassStatus.RUNNING,
    DreamPassStatus.SUBMITTED,
    DreamPassStatus.APPLYING,
)


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
    summary_for_user: str = ""
    dream_session_id: str | None = None
    ingestion_drain_status: IngestionDrainStatus = IngestionDrainStatus.drained
    snapshot: DreamOperationsSnapshot | None = None


class DreamPassOperations(BaseModel):
    """The clamped operations apply was handed, then what it reported."""

    planned: DreamOperations | None = None
    applied: DreamPassApplied | None = None


class DreamPassDraft(BaseModel):
    """A new row: whose pass it is, how it runs, what started it."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    expert_id: str | None = None
    scope_key: str = Field(min_length=1)
    route: DreamPassRoute
    trigger: DreamPassTrigger
    status: DreamPassStatus = DreamPassStatus.RUNNING
    phase: DreamPassPhase = DreamPassPhase.GATHER
    started_at: datetime | None = None


class DreamPassUpdate(BaseModel):
    """What one transition writes. A ``None`` field leaves its column as it is.

    ``status`` and ``phase`` only ever move forward. ``provider_batch_id`` is
    the batch of the update's ``phase``, which it therefore needs.
    ``phase_outputs`` and ``operations`` merge into what the row holds, one
    top-level field at a time, so a transition sends only the phase or the
    part it produced.
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

    @model_validator(mode="after")
    def _batch_names_its_phase(self) -> "DreamPassUpdate":
        if self.provider_batch_id is not None and self.phase is None:
            raise ValueError("a provider batch id needs the phase it runs")
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
