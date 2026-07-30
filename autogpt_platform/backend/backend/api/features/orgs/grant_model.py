"""Pydantic request/response models for agent-graph grants (share-with-team)."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateGrantRequest(BaseModel):
    # v1 accepts TEAM only; the field exists so enabling USER/PERSONA later is
    # a validation change, not an API break.
    principal_type: str = "TEAM"
    principal_id: str = Field(..., min_length=1)
    # None pins to the graph's current active version.
    graph_version: int | None = None
    capability: str = "EXECUTE"  # VIEW or EXECUTE
    credential_mode: str = "CONSUMER"  # CONSUMER or OWNER
    follow_latest: bool = False


class GrantResponse(BaseModel):
    id: str
    agent_graph_id: str
    agent_graph_version: int
    follow_latest: bool
    principal_type: str
    principal_id: str
    capability: str
    credential_mode: str
    org_id: str
    created_by_user_id: str
    created_at: datetime

    @staticmethod
    def from_db(grant) -> "GrantResponse":
        return GrantResponse(
            id=grant.id,
            agent_graph_id=grant.agentGraphId,
            agent_graph_version=grant.agentGraphVersion,
            follow_latest=grant.followLatest,
            principal_type=grant.principalType,
            principal_id=grant.principalId,
            capability=grant.capability,
            credential_mode=grant.credentialMode,
            org_id=grant.organizationId,
            created_by_user_id=grant.createdByUserId,
            created_at=grant.createdAt,
        )


class ReceivedGrantResponse(BaseModel):
    id: str
    agent_graph_id: str
    agent_graph_version: int
    follow_latest: bool
    principal_id: str
    capability: str
    credential_mode: str
    graph_name: str | None
    graph_description: str | None
    created_at: datetime

    @staticmethod
    def from_db(grant) -> "ReceivedGrantResponse":
        return ReceivedGrantResponse(
            id=grant.id,
            agent_graph_id=grant.agentGraphId,
            agent_graph_version=grant.agentGraphVersion,
            follow_latest=grant.followLatest,
            principal_id=grant.principalId,
            capability=grant.capability,
            credential_mode=grant.credentialMode,
            graph_name=grant.AgentGraph.name if grant.AgentGraph else None,
            graph_description=(
                grant.AgentGraph.description if grant.AgentGraph else None
            ),
            created_at=grant.createdAt,
        )
