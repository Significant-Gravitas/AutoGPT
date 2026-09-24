from backend.data import credit_metadata
from backend.data.credit_metadata import CreditMetadataMarkers

# These literals describe already-persisted, unversioned ledger rows. Keep them
# independent from the current writer contract so a writer rename cannot make
# historical charges lose their classification.
HISTORICAL_MARKERS_V1 = CreditMetadataMarkers(
    reconciliation_delta_input_key="reconciled_delta",
    execution_fee_input_key="charge",
    execution_fee_input_value="Execution Cost",
    daily_reset_reason="CoPilot daily rate limit reset",
    copilot_session_prefix="copilot-session-",
)


def reader_markers() -> tuple[CreditMetadataMarkers, ...]:
    return tuple(
        dict.fromkeys((HISTORICAL_MARKERS_V1, credit_metadata.CURRENT_CREDIT_MARKERS))
    )


def copilot_session_id(execution_id: str) -> str | None:
    prefixes = sorted(
        {marker.copilot_session_prefix for marker in reader_markers()},
        key=len,
        reverse=True,
    )
    for prefix in prefixes:
        if execution_id.startswith(prefix):
            return execution_id.removeprefix(prefix)
    return None
