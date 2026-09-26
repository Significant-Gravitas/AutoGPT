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

from ._api import (
    MAX_WAIT_SECONDS,
    ConductorAgent,
    ConductorClient,
    ConductorEffort,
    clean,
)
from ._config import conductor
from ._transcript import wait_for_reply

CREDENTIALS_DESCRIPTION = "Conductor API key from app.conductor.build/users/api-keys"


class ConductorCreateWorkspaceBlock(Block):
    execution_timeout_seconds: int | None = MAX_WAIT_SECONDS + 300

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description=CREDENTIALS_DESCRIPTION
        )
        project_id: str = SchemaField(
            description="Project (repository) to open the workspace in. Find IDs "
            "with Get Account. Use this or repository_url, not both.",
            default="",
            advanced=False,
        )
        repository_url: str = SchemaField(
            description="Git repository URL to open instead of a project ID",
            default="",
        )
        message: str = SchemaField(
            description="Initial prompt for the agent. Leave empty to create an "
            "idle workspace.",
            default="",
            advanced=False,
        )
        branch: str = SchemaField(description="Branch to start from", default="")
        name: str = SchemaField(description="Workspace name", default="")
        session_name: str = SchemaField(
            description="Name of the initial agent session", default=""
        )
        agent: ConductorAgent = SchemaField(
            description="Agent for the initial session",
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
        env: dict[str, str] = SchemaField(
            description="Environment variables for the workspace",
            default_factory=dict,
        )
        restricted_access: bool = SchemaField(
            description="Restrict the workspace to its creator", default=False
        )
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
        workspace_id: str = SchemaField(description="ID of the new workspace")
        session_id: str = SchemaField(description="ID of the initial session")
        deep_link: str = SchemaField(description="Link that opens the workspace")
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
            id="d419c562-873c-4a67-b83f-9bcf11254f3f",
            description="Create a Conductor cloud workspace for a project or "
            "repository, optionally start its agent with a prompt and wait for "
            "the reply.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "project_id": "proj_1",
                    "message": "Fix the login bug",
                },
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "project_id": "proj_1",
                    "message": "Fix the login bug",
                    "wait_for_reply": True,
                },
            ],
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("workspace_id", "ws_1"),
                ("session_id", "sess_1"),
                ("deep_link", "conductor://workspace/ws_1"),
                ("initial_message_id", "msg_1"),
                ("workspace_id", "ws_1"),
                ("session_id", "sess_1"),
                ("deep_link", "conductor://workspace/ws_1"),
                ("initial_message_id", "msg_1"),
                ("session_status", "idle"),
                ("reply", "Done, the fix is on branch fix-login."),
                ("messages", lambda m: len(m) == 1),
                ("timed_out", False),
                ("error_message", ""),
            ],
            test_mock={
                "_create": lambda *args, **kwargs: {
                    "workspaceId": "ws_1",
                    "sessionId": "sess_1",
                    "deepLink": "conductor://workspace/ws_1",
                    "initialMessage": {
                        "messageId": "msg_1",
                        "state": "queued",
                        "deepLink": "conductor://m/1",
                    },
                },
                "_wait": lambda *args, **kwargs: {
                    "session_status": "idle",
                    "error_message": "",
                    "messages": [
                        {
                            "id": "msg_2",
                            "type": "assistant",
                            "content": "Done, the fix is on branch fix-login.",
                        }
                    ],
                    "reply": "Done, the fix is on branch fix-login.",
                    "timed_out": False,
                },
            },
        )

    async def _create(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await ConductorClient(credentials).create_workspace(payload)

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
        if bool(input_data.project_id) == bool(input_data.repository_url):
            raise BlockInputError(
                message="Provide exactly one of project_id or repository_url",
                block_name=self.name,
                block_id=self.id,
            )

        payload = clean(
            {
                "projectId": input_data.project_id,
                "repositoryUrl": input_data.repository_url,
                "branch": input_data.branch,
                "name": input_data.name,
                "sessionName": input_data.session_name,
                "agent": input_data.agent,
                "model": input_data.model,
                "effort": input_data.effort,
                "message": input_data.message,
            }
        )
        if input_data.fast_mode:
            payload["fastMode"] = True
        if input_data.env:
            payload["env"] = dict(input_data.env)
        if input_data.restricted_access:
            payload["access"] = {"restricted": True}

        try:
            created = await self._create(credentials, payload)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Create workspace failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        session_id = str(created.get("sessionId") or "")
        initial = created.get("initialMessage") or {}
        message_id = str(initial.get("messageId") or "")
        yield "workspace_id", str(created.get("workspaceId") or "")
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
