"""Pydantic request/response models for resource transfer management."""

from datetime import datetime

from pydantic import BaseModel, Field

from backend.util.text_limits import (
    MAX_DESCRIPTION_LENGTH,
    MAX_ID_LENGTH,
    MAX_NAME_LENGTH,
)


class CreateTransferRequest(BaseModel):
    # "AgentGraph", "StoreListing"
    resource_type: str = Field(..., max_length=MAX_NAME_LENGTH)
    resource_id: str = Field(..., max_length=MAX_ID_LENGTH)
    target_organization_id: str = Field(..., max_length=MAX_ID_LENGTH)
    reason: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)


class TransferResponse(BaseModel):
    id: str
    resource_type: str
    resource_id: str
    source_organization_id: str
    target_organization_id: str
    initiated_by_user_id: str
    status: str
    source_approved_by_user_id: str | None
    target_approved_by_user_id: str | None
    completed_at: datetime | None
    reason: str | None
    created_at: datetime

    @staticmethod
    def from_db(tr) -> "TransferResponse":
        return TransferResponse(
            id=tr.id,
            resource_type=tr.resourceType,
            resource_id=tr.resourceId,
            source_organization_id=tr.sourceOrganizationId,
            target_organization_id=tr.targetOrganizationId,
            initiated_by_user_id=tr.initiatedByUserId,
            status=tr.status,
            source_approved_by_user_id=tr.sourceApprovedByUserId,
            target_approved_by_user_id=tr.targetApprovedByUserId,
            completed_at=tr.completedAt,
            reason=tr.reason,
            created_at=tr.createdAt,
        )
