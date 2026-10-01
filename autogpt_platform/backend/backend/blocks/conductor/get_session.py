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

from ._api import (
    DEFAULT_WAIT_SECONDS,
    MAX_WAIT_SECONDS,
    POLL_INTERVAL_DESCRIPTION,
    ConductorClient,
    poll_interval_for,
)
from ._config import conductor
from ._mocks import MOCK_PROMPT_MESSAGE, MOCK_REPLY_MESSAGE
from ._paging import fetch_after, fetch_latest_after, fetch_tail
from ._transcript import latest_reply, wait_until_idle

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorGetSessionBlock(Block):
    execution_timeout_seconds: int | None = MAX_WAIT_SECONDS + 300

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
        wait_until_idle: bool = SchemaField(
            description="Block until the session is idle or errored before "
            "reading it, instead of polling from outside. Use this to keep "
            "waiting after a Send Message or Create Session wait timed out: "
            "pass its next_after as after (the newest messages following it "
            "are returned) and the prompt's message id as prompt_message_id. "
            "Returns at once when the session is already idle.",
            default=False,
            advanced=False,
        )
        prompt_message_id: str = SchemaField(
            description="With wait_until_idle: the message_id (or "
            "initial_message_id) of the prompt being waited for. Idle is then "
            "accepted only once that prompt's turn has produced agent output, "
            "so a session that is idle because the prompt is still queued keeps "
            "being waited on.",
            default="",
            advanced=False,
        )
        timeout_seconds: int = SchemaField(
            description="How long wait_until_idle waits, in seconds (max "
            f"{MAX_WAIT_SECONDS}). Coding agents typically run 5-30 minutes: "
            "prefer a single long wait over repeated short polls or scheduled "
            "follow-ups; if timed_out is true, call again with after=next_after "
            "to keep waiting.",
            default=DEFAULT_WAIT_SECONDS,
            ge=1,
            le=MAX_WAIT_SECONDS,
        )
        poll_interval_seconds: int = SchemaField(
            description=POLL_INTERVAL_DESCRIPTION,
            default=0,
            ge=0,
            le=300,
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
            "slice: older ones by default, newer ones when after is set, and "
            "with wait_until_idle older ones following after"
        )
        next_after: str = SchemaField(
            description="ID of the last returned message; pass it as after to read "
            "what follows, or to continue a wait that timed out"
        )
        timed_out: bool = SchemaField(
            description="True when wait_until_idle was set and the session was "
            "still working when timeout_seconds elapsed; call again with "
            "after=next_after to keep waiting"
        )
        message: dict = SchemaField(
            description="The single message requested by message_id"
        )
        deep_link: str = SchemaField(description="Link that opens the session")

    def __init__(self):
        super().__init__(
            id="fabe3db0-fb32-4b05-8e68-b169477b02ec",
            description="Get a Conductor agent session: its details, whether the "
            "agent is idle, working or errored, and recent transcript messages. "
            "With wait_until_idle it blocks in-tool until the agent finishes, "
            "which is how to keep waiting after a Send Message or Create "
            "Session wait timed out (after=next_after, prompt_message_id=the "
            "prompt's message id).",
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
                ("timed_out", False),
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
        result: dict[str, Any] = {}
        if input_data.wait_until_idle:
            result["status"], result["timed_out"] = await wait_until_idle(
                client,
                input_data.session_id,
                input_data.timeout_seconds,
                poll_interval_for(
                    input_data.timeout_seconds, input_data.poll_interval_seconds
                ),
                input_data.prompt_message_id,
            )
        else:
            result["status"] = await client.session_status(input_data.session_id)
        result["session"] = await client.get_session(input_data.session_id)
        if input_data.message_limit > 0:
            if input_data.after and input_data.wait_until_idle:
                rows, has_more = await fetch_latest_after(
                    client,
                    input_data.session_id,
                    input_data.after,
                    input_data.message_limit,
                )
            elif input_data.after:
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
        yield "next_after", (
            str(messages[-1].get("id") or input_data.after)
            if messages
            else input_data.after
        )
        yield "timed_out", bool(data.get("timed_out", False))
        if data.get("message"):
            yield "message", data["message"]
        yield "deep_link", str(session.get("deepLink") or "")
