"""One ``DreamPass`` transition as a single conditional ``UPDATE``.

A pass's writes can land out of order: the batch path's submit write can reach
the database after the first result callback's, and a write the store gave up
on (``copilot/dream/store.py`` bounds each one) can still be executed by the
DatabaseManager later. So each column only moves forward, decided inside the
statement against the row as it stands when the statement runs:

  * ``status`` and ``phase`` take the new value only when it comes later in
    their enum's declared order, which is the order a pass moves through them;
  * ``providerBatchId`` is the batch of the update's ``phase``: it is written
    only while the row has not moved past that phase, so a late write for an
    earlier phase never replaces a later phase's batch;
  * ``phaseOutputs`` and ``operations`` merge key by key with jsonb ``||``, so
    two callbacks writing different phases both land;
  * ``cancelGeneration`` goes up by one when the update stops the pass from
    outside, in the same statement that closes the row, so of two stops
    racing for one open row exactly one lands and bumps it;
  * ``leaseToken``, ``leaseExpiresAt`` and ``inputBundle`` become NULL when
    the update names them in ``clear`` (a closing row drops its lease and its
    bundle);
  * every other column takes the new value when the update gives one.

Nothing is written to a row that has reached a terminal status, nor to one
that fails the update's own conditions: another user's row when it names the
owner, a row written since the instant it names. The statement text is
constant; every value is a bound parameter.
"""

from typing import Any

from pydantic import BaseModel

from backend.copilot.dream.input_bundle import input_bundle_to_dict
from backend.data.dream_pass_models import (
    OPEN_STATUSES,
    ClearableColumn,
    DreamPassUpdate,
)
from backend.util.json import dumps, sanitize_json, sanitize_string

# The column each clearable field names in ``TRANSITION_SQL``.
_CLEARABLE_COLUMNS: dict[ClearableColumn, str] = {
    "lease_token": "leaseToken",
    "lease_expires_at": "leaseExpiresAt",
    "input_bundle": "inputBundle",
}

# ``{schema_prefix}`` is filled in by ``db.execute_raw_with_schema``; the
# doubled braces are a literal empty JSON object once it has.
TRANSITION_SQL = """
UPDATE {schema_prefix}"DreamPass" SET
    "status" = CASE WHEN $2::{schema_prefix}"DreamPassStatus" > "status"
        THEN $2::{schema_prefix}"DreamPassStatus" ELSE "status" END,
    "phase" = CASE WHEN $3::{schema_prefix}"DreamPassPhase" > "phase"
        THEN $3::{schema_prefix}"DreamPassPhase" ELSE "phase" END,
    "providerBatchId" = CASE WHEN $4::text IS NOT NULL
        AND "phase" <= $3::{schema_prefix}"DreamPassPhase"
        THEN $4::text ELSE "providerBatchId" END,
    "skipReason" = COALESCE($5::text, "skipReason"),
    "error" = COALESCE($6::text, "error"),
    "leaseToken" = CASE WHEN 'leaseToken' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($7::text, "leaseToken") END,
    "leaseExpiresAt" = CASE WHEN 'leaseExpiresAt' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($8::timestamptz AT TIME ZONE 'UTC', "leaseExpiresAt") END,
    "windowStart" = COALESCE($9::timestamptz AT TIME ZONE 'UTC', "windowStart"),
    "windowEnd" = COALESCE($10::timestamptz AT TIME ZONE 'UTC', "windowEnd"),
    "submittedAt" = COALESCE($11::timestamptz AT TIME ZONE 'UTC', "submittedAt"),
    "appliedAt" = COALESCE($12::timestamptz AT TIME ZONE 'UTC', "appliedAt"),
    "completedAt" = COALESCE($13::timestamptz AT TIME ZONE 'UTC', "completedAt"),
    "inputBundle" = CASE WHEN 'inputBundle' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($14::jsonb, "inputBundle") END,
    "usage" = COALESCE($15::jsonb, "usage"),
    "phaseOutputs" = CASE WHEN $16::jsonb IS NULL THEN "phaseOutputs"
        ELSE COALESCE("phaseOutputs", '{{}}'::jsonb) || $16::jsonb END,
    "operations" = CASE WHEN $17::jsonb IS NULL THEN "operations"
        ELSE COALESCE("operations", '{{}}'::jsonb) || $17::jsonb END,
    "cancelGeneration" = "cancelGeneration"
        + CASE WHEN $19::boolean THEN 1 ELSE 0 END,
    "updatedAt" = CURRENT_TIMESTAMP AT TIME ZONE 'UTC'
WHERE "id" = $1 AND "status"::text = ANY($18::text[])
    AND ($20::text IS NULL OR "userId" = $20::text)
    AND ($21::timestamptz IS NULL
        OR "updatedAt" <= $21::timestamptz AT TIME ZONE 'UTC')
"""


def transition_args(pass_id: str, update: DreamPassUpdate) -> list[Any]:
    """``TRANSITION_SQL``'s parameters in order, ``None`` for each column the
    update leaves alone. Free text and JSON lose the control characters
    Postgres rejects."""
    return [
        pass_id,
        update.status.value if update.status is not None else None,
        update.phase.value if update.phase is not None else None,
        update.provider_batch_id,
        _text(update.skip_reason),
        _text(update.error),
        update.lease_token,
        update.lease_expires_at,
        update.window_start,
        update.window_end,
        update.submitted_at,
        update.applied_at,
        update.completed_at,
        (
            _json(input_bundle_to_dict(update.input_bundle))
            if update.input_bundle is not None
            else None
        ),
        _json(update.usage.model_dump(mode="json")) if update.usage else None,
        _merge_part(update.phase_outputs),
        _merge_part(update.operations),
        [status.value for status in OPEN_STATUSES],
        update.bump_cancel_generation,
        update.owner_user_id,
        update.not_updated_since,
        sorted(_CLEARABLE_COLUMNS[column] for column in update.clear),
    ]


def _merge_part(part: BaseModel | None) -> str | None:
    """The top-level fields *part* sets, as the JSON object merged into its
    column; ``None`` when the update leaves the column alone."""
    if part is None:
        return None
    fields = part.model_dump(mode="json")
    return _json({key: value for key, value in fields.items() if value is not None})


def _text(value: str | None) -> str | None:
    return sanitize_string(value) if value is not None else None


def _json(value: Any) -> str:
    return dumps(sanitize_json(value))
