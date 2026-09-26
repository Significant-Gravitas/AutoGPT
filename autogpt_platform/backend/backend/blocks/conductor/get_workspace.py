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

from ._api import PAGE_SIZE, ConductorClient
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
        session_limit: int = SchemaField(
            description="Maximum number of sessions to return",
            default=50,
            ge=1,
            le=500,
        )
        session_offset: int = SchemaField(
            description="Number of sessions to skip; use next_session_offset to "
            "read the following page",
            default=0,
            ge=0,
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
        sessions_has_more: bool = SchemaField(
            description="True when the workspace has more sessions than returned"
        )
        next_session_offset: int = SchemaField(
            description="session_offset to request the next page of sessions"
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
                ("sessions_has_more", False),
                ("next_session_offset", 1),
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
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        workspace_id = input_data.workspace_id
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
            result["sessions"] = await _list_sessions(client, input_data)
        except ValueError:
            result["sessions"] = {}
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self._fetch(credentials, input_data)
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
        sessions = list((data.get("sessions") or {}).get("data") or [])
        yield "sessions", sessions
        yield "sessions_has_more", bool(
            (data.get("sessions") or {}).get("hasMore", False)
        )
        yield "next_session_offset", input_data.session_offset + len(sessions)
        yield "deep_link", str(workspace.get("deepLink") or "")


async def _list_sessions(
    client: ConductorClient, input_data: ConductorGetWorkspaceBlock.Input
) -> dict[str, Any]:
    """Page the session listing (PAGE_SIZE rows per request) up to the
    requested limit and report whether more sessions follow."""
    sessions: list[dict[str, Any]] = []
    offset = input_data.session_offset
    has_more = True
    while has_more and len(sessions) < input_data.session_limit:
        page = await client.workspace_sessions(
            input_data.workspace_id,
            input_data.include_archived_sessions,
            limit=min(PAGE_SIZE, input_data.session_limit - len(sessions)),
            offset=offset,
        )
        data = list(page.get("data") or [])
        sessions.extend(data)
        offset += len(data)
        has_more = bool(page.get("hasMore")) and bool(data)
    return {"data": sessions, "hasMore": has_more}
