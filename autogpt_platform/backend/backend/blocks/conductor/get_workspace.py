from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import ConductorClient
from ._config import conductor

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorGetWorkspaceBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        workspace_id: str = SchemaField(description="Workspace ID")
        include_archived_sessions: bool = SchemaField(
            description="Include archived sessions in the session list",
            default=False,
        )

    class Output(BlockSchemaOutput):
        workspace: dict = SchemaField(
            description="Workspace: id, projectId, name, state, repoUrl, deepLink, "
            "creatorName, lastActivityAt"
        )
        status: str = SchemaField(
            description="initializing, ready, sleeping, archived, deleted, "
            "updating or unstarted"
        )
        lifecycle_step: str = SchemaField(
            description="Setup step while initializing: building_snapshot, "
            "preparing, setting_up or updating"
        )
        error_message: str = SchemaField(description="Workspace error, if any")
        preview_url: str = SchemaField(
            description="Public preview URL when a port is shared, else empty"
        )
        preview_port: int = SchemaField(
            description="Port being shared at the preview URL, 0 when none"
        )
        sessions: list[dict] = SchemaField(
            description="Agent sessions in the workspace: id, name, model, "
            "effort, deepLink"
        )
        deep_link: str = SchemaField(description="Link that opens the workspace")

    def __init__(self):
        super().__init__(
            id="c62c7983-e5c5-4468-b551-c98068610e24",
            description="Get everything about one Conductor workspace: details, "
            "current status, shared preview URL and its agent sessions.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "workspace_id": "ws_1",
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("workspace", lambda w: w["id"] == "ws_1"),
                ("status", "ready"),
                ("lifecycle_step", ""),
                ("error_message", ""),
                ("preview_url", "https://ws-1.preview.conductor.build"),
                ("preview_port", 3000),
                ("sessions", lambda s: len(s) == 1),
                ("deep_link", "conductor://workspace/ws_1"),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: {
                    "workspace": {
                        "id": "ws_1",
                        "name": "fix-login",
                        "state": "ready",
                        "repoUrl": "https://github.com/x/y",
                        "createdAt": "2026-09-26T00:00:00Z",
                        "deepLink": "conductor://workspace/ws_1",
                    },
                    "status": {
                        "workspaceId": "ws_1",
                        "status": "ready",
                        "updatedAt": "2026-09-26T00:00:00Z",
                    },
                    "preview": {
                        "preview": {
                            "port": 3000,
                            "url": "https://ws-1.preview.conductor.build",
                        }
                    },
                    "sessions": {
                        "data": [{"id": "sess_1", "deepLink": "conductor://s/1"}],
                        "offset": 0,
                        "hasMore": False,
                    },
                }
            },
        )

    async def _fetch(
        self, credentials: APIKeyCredentials, workspace_id: str, include_archived: bool
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        result: dict[str, Any] = {
            "workspace": await client.get_workspace(workspace_id),
            "status": await client.workspace_status(workspace_id),
        }
        # Preview and session listings are best-effort: a workspace that is
        # still initializing may not serve them yet.
        try:
            result["preview"] = await client.get_preview(workspace_id)
        except ValueError:
            result["preview"] = {}
        try:
            result["sessions"] = await client.workspace_sessions(
                workspace_id, include_archived
            )
        except ValueError:
            result["sessions"] = {}
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self._fetch(
                credentials,
                input_data.workspace_id,
                input_data.include_archived_sessions,
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Get workspace failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        workspace = data.get("workspace") or {}
        status = data.get("status") or {}
        preview = (data.get("preview") or {}).get("preview") or {}
        yield "workspace", workspace
        yield "status", str(status.get("status") or workspace.get("state") or "")
        yield "lifecycle_step", str(
            status.get("lifecycleStep") or workspace.get("lifecycleStep") or ""
        )
        yield "error_message", str(status.get("errorMessage") or "")
        yield "preview_url", str(preview.get("url") or "")
        yield "preview_port", int(preview.get("port") or 0)
        yield "sessions", list((data.get("sessions") or {}).get("data") or [])
        yield "deep_link", str(workspace.get("deepLink") or "")
