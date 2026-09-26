"""The durable record of a dream pass: one ``DreamPass`` row per pass.

``backend/copilot/dream/store.py`` writes the row as a pass moves, on both
routes, through ``db_accessors.dream_db()``: the scheduler and the batch
executor keep no Prisma connection, so their writes cross the DatabaseManager
RPC. The JSON columns hold the dream's own models; the input bundle keeps the
format the batch path persists to Redis (``copilot/dream/input_bundle.py``).

A pass that has reached a terminal status (complete, errored, cancelled,
expired, skipped) is final: ``update_dream_pass`` only writes a row that is
still open, the way ``job_status.mark_complete`` never rewrites a finished job.
"""

from datetime import datetime
from typing import Any

import prisma.models
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import BaseModel, Field

from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.input_bundle import (
    input_bundle_from_dict,
    input_bundle_to_dict,
)
from backend.copilot.dream.schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassUsage,
    IngestionDrainStatus,
    RecombinationOutput,
)
from backend.util.json import SafeJson, sanitize_string

OPEN_STATUSES: tuple[DreamPassStatus, ...] = (
    DreamPassStatus.QUEUED,
    DreamPassStatus.RUNNING,
    DreamPassStatus.SUBMITTED,
    DreamPassStatus.APPLYING,
)

# DreamPassUpdate's plain columns, by field name, and the free-text ones among
# them (Postgres rejects some control characters in text).
_COLUMNS: dict[str, str] = {
    "status": "status",
    "phase": "phase",
    "skip_reason": "skipReason",
    "provider_batch_id": "providerBatchId",
    "lease_token": "leaseToken",
    "lease_expires_at": "leaseExpiresAt",
    "window_start": "windowStart",
    "window_end": "windowEnd",
    "error": "error",
    "submitted_at": "submittedAt",
    "applied_at": "appliedAt",
    "completed_at": "completedAt",
}
_TEXT_COLUMNS = ("skipReason", "error")


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

    ``phase_outputs`` and ``operations`` merge into what the row holds, one
    top-level field at a time, so a transition sends only the phase or the part
    it produced.
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


async def create_dream_pass(draft: DreamPassDraft) -> DreamPassRecord:
    row = await prisma.models.DreamPass.prisma().create(
        data={
            "id": draft.id,
            "userId": draft.user_id,
            "expertId": draft.expert_id,
            "scopeKey": draft.scope_key,
            "route": draft.route,
            "trigger": draft.trigger,
            "status": draft.status,
            "phase": draft.phase,
            "startedAt": draft.started_at,
        }
    )
    return DreamPassRecord.from_db(row)


async def update_dream_pass(pass_id: str, update: DreamPassUpdate) -> bool:
    """Write *update* to the pass's row while the pass is still open.

    ``False`` when there is no such row, or it has already reached a terminal
    status. The merged JSON columns are read and written back: the steps of one
    pass run one after another, so nothing else writes the row in between.
    """
    data = _column_data(update)
    if update.phase_outputs is not None or update.operations is not None:
        current = await prisma.models.DreamPass.prisma().find_first(
            where=_open_row(pass_id)
        )
        if current is None:
            return False
        data.update(_merged_json(current, update))
    written = await prisma.models.DreamPass.prisma().update_many(
        where=_open_row(pass_id), data=data
    )
    return written > 0


async def get_dream_pass(pass_id: str) -> DreamPassRecord | None:
    row = await prisma.models.DreamPass.prisma().find_unique(where={"id": pass_id})
    return DreamPassRecord.from_db(row) if row else None


async def get_dream_pass_for_user(pass_id: str, user_id: str) -> DreamPassRecord | None:
    """The pass's row when *user_id* owns it. ``None`` for another user's pass,
    the same as for a missing one, so a caller cannot probe for pass ids."""
    row = await prisma.models.DreamPass.prisma().find_first(
        where={"id": pass_id, "userId": user_id}
    )
    return DreamPassRecord.from_db(row) if row else None


async def list_open_dream_passes(scope_key: str) -> list[DreamPassRecord]:
    """The scope's passes that have not reached a terminal status, oldest first."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={"scopeKey": scope_key, "status": {"in": list(OPEN_STATUSES)}},
        order={"createdAt": "asc"},
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_dream_passes(user_id: str, limit: int = 20) -> list[DreamPassRecord]:
    """The user's passes, across every scope, newest first."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={"userId": user_id}, order={"createdAt": "desc"}, take=limit
    )
    return [DreamPassRecord.from_db(row) for row in rows]


def _open_row(pass_id: str) -> dict[str, Any]:
    return {"id": pass_id, "status": {"in": list(OPEN_STATUSES)}}


def _column_data(update: DreamPassUpdate) -> dict[str, Any]:
    """The update's plain and replaced JSON columns, skipping ``None`` fields."""
    plain = update.model_dump(include=set(_COLUMNS), exclude_none=True)
    data: dict[str, Any] = {_COLUMNS[field]: value for field, value in plain.items()}
    for column in _TEXT_COLUMNS:
        if column in data:
            data[column] = sanitize_string(data[column])
    if update.input_bundle is not None:
        data["inputBundle"] = SafeJson(input_bundle_to_dict(update.input_bundle))
    if update.usage is not None:
        data["usage"] = SafeJson(update.usage)
    return data


def _merged_json(
    current: prisma.models.DreamPass, update: DreamPassUpdate
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if update.phase_outputs is not None:
        data["phaseOutputs"] = SafeJson(
            _merge(current.phaseOutputs, update.phase_outputs)
        )
    if update.operations is not None:
        data["operations"] = SafeJson(_merge(current.operations, update.operations))
    return data


def _merge(stored: Any, part: BaseModel) -> dict[str, Any]:
    """*stored* with each top-level field *part* sets written over it."""
    fresh = {
        field: value
        for field, value in part.model_dump(mode="json").items()
        if value is not None
    }
    return {**(dict(stored) if stored else {}), **fresh}
