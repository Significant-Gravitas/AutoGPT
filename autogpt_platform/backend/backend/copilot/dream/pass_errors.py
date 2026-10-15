"""The errors a dream pass records: ``<stage>: <code>``, both from the fixed
sets below, so no recorded error holds the model's output, the pass's input
or an exception's text (those go to the log). The store records no other
error (``data/dream_pass_update.transition_args`` turns any other into
``pass: unrecorded``), and a hard forget keeps an error only when it is one
of these (``dream_error_blanked``, in the migration that added the mark).
"""

from typing import Literal, get_args

from backend.copilot.inference.context import InferenceErrorCode

# Where the pass was: the pass as a whole, its billing check, one of its
# steps (the row's phases), the batch route's handoff and its callbacks'
# payload, or a stop from outside (a cancel, an expiry).
PassErrorStage = Literal[
    "pass",
    "billing",
    "gather",
    "consolidate",
    "recombine",
    "sanitize",
    "apply",
    "done",
    "submit",
    "batch",
    "cancelled",
    "expired",
]

# What went wrong: a phase's answer (``InferenceError.code``), the
# provider's, the infrastructure, the pass's lock or record, a batch
# payload, or why a stop from outside ended it.
PassErrorCode = Literal[
    InferenceErrorCode,
    "provider_error",
    "no_rows",
    "missing_content",
    "crashed",
    "import_failed",
    "no_api_key",
    "submit_failed",
    "input_persist_failed",
    "gate_unavailable",
    "rate_limit_unavailable",
    "charge_failed",
    "record_unconfirmed",
    "record_failed",
    "unrecorded",
    "not_on_record",
    "lock_lost",
    "lease_unrenewed",
    "row_refused",
    "payload_incomplete",
    "usage_attribution_unavailable",
    "unknown_phase",
    "input_missing",
    "scope_mismatch",
    "already_closed",
    "duplicate_delivery",
    "by_admin",
    "by_wipe",
    "forced",
    "lease_lapsed",
    "stalled",
    "reaped",
    "stopped",
]

PASS_ERROR_STAGES: tuple[PassErrorStage, ...] = get_args(PassErrorStage)
PASS_ERROR_CODES: tuple[PassErrorCode, ...] = get_args(PassErrorCode)

# What the store records in place of an error that is not one of these.
UNRECORDED = "pass: unrecorded"


def pass_error(stage: PassErrorStage, code: PassErrorCode) -> str:
    """The error a pass records: ``<stage>: <code>``."""
    return f"{stage}: {code}"


def is_pass_error(error: str) -> bool:
    """Whether *error* is ``<stage>: <code>`` from the fixed sets."""
    stage, separator, code = error.partition(": ")
    return bool(separator) and stage in PASS_ERROR_STAGES and code in PASS_ERROR_CODES
