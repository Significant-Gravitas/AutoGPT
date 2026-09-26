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
from ._mocks import MOCK_PROMPT_MESSAGE, MOCK_REPLY_MESSAGE
from ._paging import fetch_after, fetch_tail
from ._transcript import latest_reply

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorGetSessionBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        session_id: str = SchemaField(description="Session ID")
        message_limit: int = SchemaField(
            description="How many transcript messages to return: the most recent "
            "ones, or the ones following `after` when it is set; 0 skips the "
            "transcript",
            default=20,
            ge=0,
            le=500,
            advanced=False,
        )
        after: str = SchemaField(
            description="Read forward from this transcript message ID (exclusive) "
            "instead of returning the most recent messages; use next_after from "
            "a previous call to poll incrementally",
            default="",
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
            description="Transcript messages, oldest first: id, sessionIndex, "
            "type, content, receivedAt"
        )
        latest_reply: str = SchemaField(
            description="Text of the newest agent message with visible text in "
            "the returned transcript slice"
        )
        has_more: bool = SchemaField(
            description="True when the transcript has messages beyond the returned "
            "slice: older ones by default, newer ones when after is set"
        )
        next_after: str = SchemaField(
            description="ID of the last returned message; pass it as after to read "
            "what follows"
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
                ("has_more", False),
                ("next_after", "row_2"),
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
                        "data": [MOCK_PROMPT_MESSAGE, MOCK_REPLY_MESSAGE],
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
            if input_data.after:
                rows, has_more = await fetch_after(
                    client,
                    input_data.session_id,
                    input_data.after,
                    input_data.message_limit,
                )
            else:
                rows, has_more = await fetch_tail(
                    client, input_data.session_id, input_data.message_limit
                )
            result["messages"] = {"data": rows, "hasMore": has_more}
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
        listing = data.get("messages") or {}
        messages = list(listing.get("data") or [])
        yield "session", session
        yield "status", str(status.get("status") or "")
        yield "error_message", str(
            status.get("errorMessage") or status.get("lastError") or ""
        )
        yield "messages", messages
        yield "latest_reply", latest_reply(messages)
        yield "has_more", bool(listing.get("hasMore", False))
        yield "next_after", str(messages[-1].get("id") or "") if messages else ""
        if data.get("message"):
            yield "message", data["message"]
        yield "deep_link", str(session.get("deepLink") or "")
