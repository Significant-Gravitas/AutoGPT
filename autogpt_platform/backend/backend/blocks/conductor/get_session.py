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
from ._mocks import WAIT_MOCK_REPLY
from ._transcript import is_agent_message, message_text

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorGetSessionBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        session_id: str = SchemaField(description="Session ID")
        message_limit: int = SchemaField(
            description="How many transcript messages to return; 0 skips the "
            "transcript",
            default=20,
            ge=0,
            le=500,
            advanced=False,
        )
        after: str = SchemaField(
            description="Only messages after this message ID", default=""
        )
        message_id: str = SchemaField(
            description="Also fetch this single message by ID", default=""
        )

    class Output(BlockSchemaOutput):
        session: dict = SchemaField(
            description="Session: id, name, model, resolvedModel, effort, "
            "fastMode, deepLink, archivedAt"
        )
        status: str = SchemaField(description="idle, working or error")
        error_message: str = SchemaField(description="Last session error, if any")
        messages: list[dict] = SchemaField(
            description="Transcript messages: id, sessionIndex, type, content, "
            "receivedAt"
        )
        latest_reply: str = SchemaField(
            description="Text of the most recent agent message in the returned "
            "transcript slice"
        )
        message: dict = SchemaField(
            description="The single message requested by message_id"
        )
        deep_link: str = SchemaField(description="Link that opens the session")

    def __init__(self):
        super().__init__(
            id="fabe3db0-fb32-4b05-8e68-b169477b02ec",
            description="Get a Conductor agent session: its details, whether the "
            "agent is idle, working or errored, and recent transcript messages.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "session_id": "sess_1",
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("session", lambda s: s["id"] == "sess_1"),
                ("status", "idle"),
                ("error_message", ""),
                ("messages", lambda m: len(m) == 2),
                ("latest_reply", "All tests pass now."),
                ("deep_link", "conductor://s/1"),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: {
                    "session": {"id": "sess_1", "deepLink": "conductor://s/1"},
                    "status": {
                        "workspaceId": "ws_1",
                        "sessionId": "sess_1",
                        "status": "idle",
                        "updatedAt": "2026-09-26T00:00:00Z",
                    },
                    "messages": {
                        "data": [
                            {"id": "msg_1", "type": "user", "content": "Run tests"},
                            WAIT_MOCK_REPLY["messages"][0],
                        ],
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
        result: dict[str, Any] = {
            "session": await client.get_session(input_data.session_id),
            "status": await client.session_status(input_data.session_id),
        }
        if input_data.message_limit > 0:
            result["messages"] = await client.list_messages(
                input_data.session_id,
                after=input_data.after,
                limit=input_data.message_limit,
            )
        if input_data.message_id:
            result["message"] = await client.get_message(input_data.message_id)
        return result

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self._fetch(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Get session failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        session = data.get("session") or {}
        status = data.get("status") or {}
        messages = list((data.get("messages") or {}).get("data") or [])
        agent_messages = [m for m in messages if is_agent_message(m)]
        yield "session", session
        yield "status", str(status.get("status") or "")
        yield "error_message", str(
            status.get("errorMessage") or status.get("lastError") or ""
        )
        yield "messages", messages
        yield "latest_reply", message_text(agent_messages[-1]) if agent_messages else ""
        if data.get("message"):
            yield "message", data["message"]
        yield "deep_link", str(session.get("deepLink") or "")
