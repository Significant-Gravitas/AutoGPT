from __future__ import annotations

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class ArtifactsTable(BaseNoSQLTable):
    table_name = "artifacts"
    primary_key = "artifact_id"
    secondary_key = "agent_id"
    third_key = "user_id"

    from AFAAS.lib.sdk.artifacts import Artifact

    def add(self, value: dict, id: str = Artifact.generate_uuid()) -> str:
        return super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    def update(self, plan_id: str, agent_id: str, value: dict):
        key = ArtifactsTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().update(key=key, value=value)

    def delete(self, plan_id: str, agent_id: str):
        key = ArtifactsTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().delete(key=key)

    def get(self, plan_id: str, agent_id: str) -> Artifact:
        key = ArtifactsTable.Key(
            primary_key=str(plan_id),
            secondary_key=str(agent_id),
        )
        return super().get(key=key)
