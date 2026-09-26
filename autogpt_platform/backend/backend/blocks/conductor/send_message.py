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

from ._api import MAX_WAIT_SECONDS, ConductorClient
from ._config import conductor
from ._mocks import WAIT_MOCK_REPLY
from ._transcript import wait_for_reply

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorSendMessageBlock(Block):
    execution_timeout_seconds: int | None = MAX_WAIT_SECONDS + 300

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        session_id: str = SchemaField(description="Session to send the prompt to")
        message: str = SchemaField(description="Prompt for the agent")
        wait_for_reply: bool = SchemaField(
            description="Wait until the agent is idle and return its reply",
            default=True,
            advanced=False,
        )
        timeout_seconds: int = SchemaField(
            description="How long to wait for the reply",
            default=900,
            ge=1,
            le=MAX_WAIT_SECONDS,
        )
        poll_interval_seconds: int = SchemaField(
            description="Seconds between status checks while waiting",
            default=10,
            ge=1,
            le=300,
        )

    class Output(BlockSchemaOutput):
        message_id: str = SchemaField(description="ID of the sent prompt")
        state: str = SchemaField(description="queued or sent")
        deep_link: str = SchemaField(description="Link that opens the message")
        session_status: str = SchemaField(
            description="idle, working or error once waiting finished"
        )
        reply: str = SchemaField(description="Text the agent produced in response")
        messages: list[dict] = SchemaField(
            description="Raw transcript messages after the prompt"
        )
        timed_out: bool = SchemaField(
            description="True when the wait ended before the agent went idle"
        )
        truncated: bool = SchemaField(
            description="True when the turn produced more messages than are "
            "kept; messages holds the newest ones and reply may be incomplete"
        )
        error_message: str = SchemaField(description="Session error, if any")

    def __init__(self):
        super().__init__(
            id="4f22dbe5-2855-498a-ab82-6857441ea4b1",
            description="Send a prompt to a Conductor agent session and, by "
            "default, wait for the agent to finish and return its reply.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": conductor.get_test_credentials().model_dump(),
                "session_id": "sess_1",
                "message": "Run the tests",
            },
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("message_id", "msg_1"),
                ("state", "queued"),
                ("deep_link", "conductor://m/1"),
                ("session_status", "idle"),
                ("reply", "All tests pass now."),
                ("messages", lambda m: len(m) == 1),
                ("timed_out", False),
                ("truncated", False),
                ("error_message", ""),
            ],
            test_mock={
                "_send": lambda *args, **kwargs: {
                    "messageId": "msg_1",
                    "state": "queued",
                    "deepLink": "conductor://m/1",
                },
                "_wait": lambda *args, **kwargs: WAIT_MOCK_REPLY,
            },
        )

    async def _send(
        self, credentials: APIKeyCredentials, session_id: str, message: str
    ) -> dict[str, Any]:
        return await ConductorClient(credentials).send_message(session_id, message)

    async def _wait(
        self,
        credentials: APIKeyCredentials,
        session_id: str,
        after_message_id: str,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> dict[str, Any]:
        return await wait_for_reply(
            ConductorClient(credentials),
            session_id,
            after_message_id,
            timeout_seconds,
            poll_interval_seconds,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.message.strip():
            raise BlockInputError(
                message="message must not be empty",
                block_name=self.name,
                block_id=self.id,
            )

        try:
            sent = await self._send(
                credentials, input_data.session_id, input_data.message
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Send message failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        message_id = str(sent.get("messageId") or "")
        yield "message_id", message_id
        yield "state", str(sent.get("state") or "")
        yield "deep_link", str(sent.get("deepLink") or "")

        if not input_data.wait_for_reply:
            return
        if not message_id:
            raise BlockExecutionError(
                message="Cannot wait for the agent: Conductor returned no messageId",
                block_name=self.name,
                block_id=self.id,
            )
        try:
            waited = await self._wait(
                credentials,
                input_data.session_id,
                message_id,
                input_data.timeout_seconds,
                input_data.poll_interval_seconds,
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Waiting for the agent failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e
        yield "session_status", waited["session_status"]
        yield "reply", waited["reply"]
        yield "messages", waited["messages"]
        yield "timed_out", waited["timed_out"]
        yield "truncated", bool(waited.get("truncated", False))
        yield "error_message", waited["error_message"]
