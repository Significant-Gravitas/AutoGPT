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
  * ``readUuids`` gains the update's, never losing one;
  * ``leaseExpiresAt`` only moves later: a renewal written late (a sync
    renewal the store gave up on, landing after the batch submit's longer
    lease) never shortens the lease the row holds. Every renewal sets an
    expiry at the renewal time plus its route's TTL;
  * ``leaseToken``, ``leaseExpiresAt``, ``inputBundle`` and
    ``cleanupPendingAt`` become NULL when the update names them in ``clear``
    (a closing row drops its bundle, and its lease unless the close marks
    it for a cleanup, whose end then clears the lease with the mark): the
    only way a lease is dropped or cut short;
  * every other column takes the new value when the update gives one.

A row a hard forget marked (``erasedAt``, ``copilot/dream/forget_copies.py``)
has the dream's text blanked by the update that closes it, in the same
statement, whichever way it closes: the pass's own end, a cancel, an expiry,
the reaper's close, and again by the cleanup's end. Every string under one of
``DREAM_TEXT_FIELDS`` in its phase outputs and operations, the update's own
included, reads ``ERASED_TEXT`` (``dream_text_blanked``, a function of the
migration that added the mark), and its error is kept only when it is one of
the fixed codes (``dream_error_blanked``). The mark itself is the forget's to
write and nothing clears it, so a pass that was still running when it landed
is blanked however late it ends.

Every error the row takes is one of the fixed codes
(``copilot/dream/pass_errors.py``): any other is recorded as
``pass: unrecorded``, logged, so no exception's or model's text reaches the
row.

Nothing is written to a row that has reached a terminal status, nor to one
that fails the update's own conditions: another user's row when it names the
owner, a row written since the instant it names. The one exception is the
update marked ``closed_row`` (the cleanup after the pass saying it has
finished), which applies to a closed row only and only clears. The statement
text is constant; every value is a bound parameter.
"""

import logging
from typing import Any

from pydantic import BaseModel

from backend.copilot.dream.input_bundle import input_bundle_to_dict
from backend.copilot.dream.pass_errors import (
    PASS_ERROR_CODES,
    PASS_ERROR_STAGES,
    UNRECORDED,
    is_pass_error,
)
from backend.data.dream_pass_models import (
    CLOSED_STATUSES,
    DREAM_TEXT_FIELDS,
    ERASED_TEXT,
    OPEN_STATUSES,
    ClearableColumn,
    DreamPassUpdate,
)
from backend.util.json import dumps, sanitize_json, sanitize_string

logger = logging.getLogger(__name__)

# The column each clearable field names in ``TRANSITION_SQL``.
_CLEARABLE_COLUMNS: dict[ClearableColumn, str] = {
    "lease_token": "leaseToken",
    "lease_expires_at": "leaseExpiresAt",
    "input_bundle": "inputBundle",
    "cleanup_pending_at": "cleanupPendingAt",
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
    "error" = CASE
        WHEN "erasedAt" IS NOT NULL AND $24::boolean
        THEN {schema_prefix}"dream_error_blanked"(
            COALESCE($6::text, "error"), $28::text[], $29::text[], $26::text)
        ELSE COALESCE($6::text, "error") END,
    "leaseToken" = CASE WHEN 'leaseToken' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($7::text, "leaseToken") END,
    "leaseExpiresAt" = CASE WHEN 'leaseExpiresAt' = ANY($22::text[]) THEN NULL
        ELSE GREATEST($8::timestamptz AT TIME ZONE 'UTC', "leaseExpiresAt") END,
    "windowStart" = COALESCE($9::timestamptz AT TIME ZONE 'UTC', "windowStart"),
    "windowEnd" = COALESCE($10::timestamptz AT TIME ZONE 'UTC', "windowEnd"),
    "readUuids" = CASE WHEN $27::text[] IS NULL THEN "readUuids"
        ELSE ARRAY(SELECT DISTINCT uuid
            FROM unnest(COALESCE("readUuids", ARRAY[]::text[]) || $27::text[])
                AS read(uuid)
            ORDER BY uuid) END,
    "submittedAt" = COALESCE($11::timestamptz AT TIME ZONE 'UTC', "submittedAt"),
    "appliedAt" = COALESCE($12::timestamptz AT TIME ZONE 'UTC', "appliedAt"),
    "completedAt" = COALESCE($13::timestamptz AT TIME ZONE 'UTC', "completedAt"),
    "inputBundle" = CASE WHEN 'inputBundle' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($14::jsonb, "inputBundle") END,
    "usage" = COALESCE($15::jsonb, "usage"),
    "phaseOutputs" = CASE
        WHEN "erasedAt" IS NOT NULL AND $24::boolean
        THEN {schema_prefix}"dream_text_blanked"(
            CASE WHEN $16::jsonb IS NULL THEN "phaseOutputs"
            ELSE COALESCE("phaseOutputs", '{{}}'::jsonb) || $16::jsonb END,
            $25::text[], $26::text)
        WHEN $16::jsonb IS NULL THEN "phaseOutputs"
        ELSE COALESCE("phaseOutputs", '{{}}'::jsonb) || $16::jsonb END,
    "operations" = CASE
        WHEN "erasedAt" IS NOT NULL AND $24::boolean
        THEN {schema_prefix}"dream_text_blanked"(
            CASE WHEN $17::jsonb IS NULL THEN "operations"
            ELSE COALESCE("operations", '{{}}'::jsonb) || $17::jsonb END,
            $25::text[], $26::text)
        WHEN $17::jsonb IS NULL THEN "operations"
        ELSE COALESCE("operations", '{{}}'::jsonb) || $17::jsonb END,
    "cancelGeneration" = "cancelGeneration"
        + CASE WHEN $19::boolean THEN 1 ELSE 0 END,
    "cleanupPendingAt" = CASE WHEN 'cleanupPendingAt' = ANY($22::text[]) THEN NULL
        ELSE COALESCE($23::timestamptz AT TIME ZONE 'UTC', "cleanupPendingAt") END,
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
        _recorded(update.error),
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
        [
            status.value
            for status in (CLOSED_STATUSES if update.closed_row else OPEN_STATUSES)
        ],
        update.bump_cancel_generation,
        update.owner_user_id,
        update.not_updated_since,
        sorted(_CLEARABLE_COLUMNS[column] for column in update.clear),
        update.cleanup_pending_at,
        update.closed_row or update.status in CLOSED_STATUSES,
        list(DREAM_TEXT_FIELDS),
        ERASED_TEXT,
        update.read_uuids,
        list(PASS_ERROR_STAGES),
        list(PASS_ERROR_CODES),
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


def _recorded(error: str | None) -> str | None:
    """*error* when it is one of the fixed codes, else ``UNRECORDED``."""
    if error is None or is_pass_error(error):
        return error
    logger.warning(
        "A dream pass error that is not one of the fixed codes is recorded as %r",
        UNRECORDED,
        stack_info=True,
    )
    return UNRECORDED


def _json(value: Any) -> str:
    return dumps(sanitize_json(value))
