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
    MAX_WAIT_SECONDS,
    ConductorAgent,
    ConductorClient,
    ConductorEffort,
    clean,
)
from ._config import conductor
from ._mocks import WAIT_MOCK_REPLY
from ._transcript import wait_for_reply

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorCreateSessionBlock(Block):
    execution_timeout_seconds: int | None = MAX_WAIT_SECONDS + 300

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        workspace_id: str = SchemaField(description="Workspace to add the session to")
        message: str = SchemaField(
            description="Initial prompt for the agent; leave empty for an idle "
            "session",
            default="",
            advanced=False,
        )
        agent: ConductorAgent = SchemaField(
            description="Agent to run",
            default=ConductorAgent.CLAUDE,
            advanced=False,
        )
        model: str = SchemaField(
            description="Model id such as fable-5-1, opus-5-5-1m, sonnet-5-1m, "
            "gpt-6-astra or auto. Leave empty for Conductor's default.",
            default="",
            advanced=False,
        )
        effort: ConductorEffort = SchemaField(
            description="Reasoning effort; leave empty for the default",
            default=ConductorEffort.DEFAULT,
        )
        fast_mode: bool = SchemaField(description="Enable fast mode", default=False)
        name: str = SchemaField(description="Session name", default="")
        wait_for_reply: bool = SchemaField(
            description="After sending the initial prompt, wait until the agent is "
            "idle and return its reply",
            default=False,
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
        session_id: str = SchemaField(description="ID of the new session")
        deep_link: str = SchemaField(description="Link that opens the session")
        initial_message_id: str = SchemaField(
            description="ID of the initial prompt message, empty when none was sent"
        )
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
        error_message: str = SchemaField(description="Session error, if any")

    def __init__(self):
        super().__init__(
            id="3d4b2ccb-d654-4b5d-9750-e67f393f53f0",
            description="Start a new agent session (chat) in an existing Conductor "
            "workspace, optionally with a first prompt, and optionally wait for "
            "the agent's reply.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "workspace_id": "ws_1",
                    "message": "Run the tests",
                },
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "workspace_id": "ws_1",
                    "message": "Run the tests",
                    "wait_for_reply": True,
                },
            ],
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("session_id", "sess_1"),
                ("deep_link", "conductor://s/1"),
                ("initial_message_id", "msg_1"),
                ("session_id", "sess_1"),
                ("deep_link", "conductor://s/1"),
                ("initial_message_id", "msg_1"),
                ("session_status", "idle"),
                ("reply", "All tests pass now."),
                ("messages", lambda m: len(m) == 1),
                ("timed_out", False),
                ("error_message", ""),
            ],
            test_mock={
                "_create": lambda *args, **kwargs: {
                    "id": "sess_1",
                    "deepLink": "conductor://s/1",
                    "initialMessage": {
                        "messageId": "msg_1",
                        "state": "queued",
                        "deepLink": "conductor://m/1",
                    },
                },
                "_wait": lambda *args, **kwargs: WAIT_MOCK_REPLY,
            },
        )

    async def _create(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await ConductorClient(credentials).create_session(payload)

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
        payload = clean(
            {
                "workspaceId": input_data.workspace_id,
                "agent": input_data.agent,
                "model": input_data.model,
                "effort": input_data.effort,
                "name": input_data.name,
                "message": input_data.message,
            }
        )
        if input_data.fast_mode:
            payload["fastMode"] = True

        try:
            created = await self._create(credentials, payload)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Create session failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        session_id = str(created.get("id") or "")
        initial = created.get("initialMessage") or {}
        message_id = str(initial.get("messageId") or "")
        yield "session_id", session_id
        yield "deep_link", str(created.get("deepLink") or "")
        yield "initial_message_id", message_id

        if not (input_data.wait_for_reply and session_id and message_id):
            return
        try:
            waited = await self._wait(
                credentials,
                session_id,
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
        yield "error_message", waited["error_message"]
