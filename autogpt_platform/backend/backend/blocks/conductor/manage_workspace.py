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
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import ConductorClient, WorkspaceAction
from ._config import conductor

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorManageWorkspaceBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        workspace_id: str = SchemaField(description="Workspace ID")
        action: WorkspaceAction = SchemaField(
            description="rename, archive, unarchive, sleep, share_preview (expose "
            "a port at a public preview URL), stop_preview, or move_to_section",
            default=WorkspaceAction.RENAME,
            advanced=False,
        )
        name: str = SchemaField(
            description="New workspace name (rename)", default="", advanced=False
        )
        port: int = SchemaField(
            description="Port inside the workspace to share (share_preview)",
            default=3000,
            ge=1,
            le=65535,
            advanced=False,
        )
        section_id: str = SchemaField(
            description="Target section ID (move_to_section); empty clears the "
            "section",
            default="",
            advanced=False,
        )

    class Output(BlockSchemaOutput):
        workspace_id: str = SchemaField(description="Workspace ID")
        status: str = SchemaField(
            description="Workspace state after the action, when reported"
        )
        preview_url: str = SchemaField(
            description="Preview URL after a share_preview action, else empty"
        )
        result: dict = SchemaField(description="Raw API response")

    def __init__(self):
        super().__init__(
            id="96582ded-a249-4998-8454-eabc8ffc28fc",
            description="Change a Conductor workspace: rename it, archive, "
            "unarchive or sleep it, share or stop sharing a port at its public "
            "preview URL, or move it into a section.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "workspace_id": "ws_1",
                    "action": WorkspaceAction.SHARE_PREVIEW.value,
                    "port": 3000,
                },
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "workspace_id": "ws_1",
                    "action": WorkspaceAction.ARCHIVE.value,
                },
            ],
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("workspace_id", "ws_1"),
                ("status", ""),
                ("preview_url", "https://ws-1.preview.conductor.build"),
                ("result", lambda r: r["preview"]["port"] == 3000),
                ("workspace_id", "ws_1"),
                ("status", "archived"),
                ("preview_url", ""),
                ("result", {"workspaceId": "ws_1", "status": "archived"}),
            ],
            test_mock={
                "_perform": lambda credentials, input_data: (
                    {
                        "preview": {
                            "port": 3000,
                            "url": "https://ws-1.preview.conductor.build",
                        }
                    }
                    if input_data.action == WorkspaceAction.SHARE_PREVIEW
                    else {"workspaceId": "ws_1", "status": "archived"}
                )
            },
        )

    async def _perform(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        ws = input_data.workspace_id
        action = input_data.action
        if action == WorkspaceAction.RENAME:
            return await client.rename_workspace(ws, input_data.name)
        if action in (
            WorkspaceAction.ARCHIVE,
            WorkspaceAction.UNARCHIVE,
            WorkspaceAction.SLEEP,
        ):
            return await client.workspace_lifecycle(ws, action.value)
        if action == WorkspaceAction.SHARE_PREVIEW:
            return await client.share_preview(ws, input_data.port)
        if action == WorkspaceAction.STOP_PREVIEW:
            return await client.stop_preview(ws)
        return await client.set_workspace_section(ws, input_data.section_id or None)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.action == WorkspaceAction.RENAME and not input_data.name.strip():
            raise BlockInputError(
                message="name is required to rename a workspace",
                block_name=self.name,
                block_id=self.id,
            )

        try:
            result = await self._perform(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Workspace {input_data.action.value} failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        preview = result.get("preview") or {}
        yield "workspace_id", input_data.workspace_id
        yield "status", str(result.get("status") or result.get("state") or "")
        yield "preview_url", str(preview.get("url") or "")
        yield "result", result
