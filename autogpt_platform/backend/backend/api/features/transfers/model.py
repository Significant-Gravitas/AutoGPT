"""Pydantic request/response models for resource transfer management."""

from datetime import datetime

from pydantic import BaseModel


class CreateTransferRequest(BaseModel):
    resource_type: str  # "AgentGraph", "StoreListing"
    resource_id: str
    target_organization_id: str
    reason: str | None = None


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
