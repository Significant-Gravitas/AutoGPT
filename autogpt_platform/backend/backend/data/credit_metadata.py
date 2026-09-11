from pydantic import BaseModel, ConfigDict


class CreditMetadataMarkers(BaseModel):
    model_config = ConfigDict(frozen=True)

    reconciliation_delta_input_key: str
    execution_fee_input_key: str
    execution_fee_input_value: str
    daily_reset_reason: str
    copilot_session_prefix: str


CURRENT_CREDIT_MARKERS = CreditMetadataMarkers(
    reconciliation_delta_input_key="reconciled_delta",
    execution_fee_input_key="charge",
    execution_fee_input_value="Execution Cost",
    daily_reset_reason="CoPilot daily rate limit reset",
    copilot_session_prefix="copilot-session-",
)
