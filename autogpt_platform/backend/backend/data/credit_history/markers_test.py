from backend.data import credit_metadata
from backend.data.credit_history.markers import (
    HISTORICAL_MARKERS_V1,
    copilot_session_id,
    reader_markers,
)


def test_reader_keeps_literal_v1_markers_when_current_contract_changes(monkeypatch):
    monkeypatch.setattr(
        credit_metadata,
        "CURRENT_CREDIT_MARKERS",
        credit_metadata.CreditMetadataMarkers(
            reconciliation_delta_input_key="reconciled_delta_v2",
            execution_fee_input_key="charge_v2",
            execution_fee_input_value="Execution Cost v2",
            daily_reset_reason="Daily reset v2",
            copilot_session_prefix="copilot-session-v2-",
        ),
    )

    assert HISTORICAL_MARKERS_V1 == credit_metadata.CreditMetadataMarkers(
        reconciliation_delta_input_key="reconciled_delta",
        execution_fee_input_key="charge",
        execution_fee_input_value="Execution Cost",
        daily_reset_reason="CoPilot daily rate limit reset",
        copilot_session_prefix="copilot-session-",
    )
    assert reader_markers() == (
        HISTORICAL_MARKERS_V1,
        credit_metadata.CURRENT_CREDIT_MARKERS,
    )


def test_copilot_session_id_prefers_longest_supported_prefix(monkeypatch):
    monkeypatch.setattr(
        credit_metadata,
        "CURRENT_CREDIT_MARKERS",
        credit_metadata.CreditMetadataMarkers(
            reconciliation_delta_input_key="reconciled_delta_v2",
            execution_fee_input_key="charge_v2",
            execution_fee_input_value="Execution Cost v2",
            daily_reset_reason="Daily reset v2",
            copilot_session_prefix="copilot-session-v2-",
        ),
    )

    assert copilot_session_id("copilot-session-v2-chat") == "chat"
    assert copilot_session_id("copilot-session-chat") == "chat"
    assert copilot_session_id("ordinary-execution") is None
