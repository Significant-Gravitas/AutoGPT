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

from ._api import ConductorClient, SessionAction
from ._config import conductor

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorManageSessionBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        session_id: str = SchemaField(description="Session ID")
        action: SessionAction = SchemaField(
            description="rename, cancel (stop the current turn and drop queued "
            "prompts) or archive",
            default=SessionAction.CANCEL,
            advanced=False,
        )
        name: str = SchemaField(
            description="New session name (rename)", default="", advanced=False
        )

    class Output(BlockSchemaOutput):
        session_id: str = SchemaField(description="Session ID")
        workspace_id: str = SchemaField(
            description="Workspace the session belongs to, when reported"
        )
        status: str = SchemaField(
            description="Session status after the action, when reported"
        )
        canceled_queued_messages: int = SchemaField(
            description="Queued prompts dropped by cancel or archive"
        )
        result: dict = SchemaField(description="Raw API response")

    def __init__(self):
        super().__init__(
            id="3923754c-2198-4925-a1eb-06757eb7c4d0",
            description="Rename, cancel or archive a Conductor agent session. "
            "Cancel stops the running turn and drops queued prompts.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "session_id": "sess_1",
                "action": SessionAction.CANCEL.value,
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("session_id", "sess_1"),
                ("workspace_id", "ws_1"),
                ("status", "idle"),
                ("canceled_queued_messages", 2),
                ("result", lambda r: r["canceledQueuedMessages"] == 2),
            ],
            test_mock={
                "_perform": lambda *args, **kwargs: {
                    "workspaceId": "ws_1",
                    "sessionId": "sess_1",
                    "status": "idle",
                    "canceledQueuedMessages": 2,
                }
            },
        )

    async def _perform(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        if input_data.action == SessionAction.RENAME:
            return await client.rename_session(input_data.session_id, input_data.name)
        if input_data.action == SessionAction.CANCEL:
            return await client.cancel_session(input_data.session_id)
        return await client.archive_session(input_data.session_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.action == SessionAction.RENAME and not input_data.name.strip():
            raise BlockInputError(
                message="name is required to rename a session",
                block_name=self.name,
                block_id=self.id,
            )

        try:
            result = await self._perform(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Session {input_data.action.value} failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "session_id", str(result.get("sessionId") or input_data.session_id)
        yield "workspace_id", str(result.get("workspaceId") or "")
        yield "status", str(result.get("status") or "")
        yield "canceled_queued_messages", int(result.get("canceledQueuedMessages") or 0)
        yield "result", result
