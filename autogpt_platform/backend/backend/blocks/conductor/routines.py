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

from ._api import ConductorAgent, ConductorClient, ConductorEffort, RoutineAction, clean
from ._config import conductor


def webhook_url(routine: dict[str, Any]) -> str:
    for trigger in routine.get("triggers") or []:
        if trigger.get("webhookUrl"):
            return str(trigger["webhookUrl"])
    return ""


class ConductorManageRoutineBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description="Conductor API key from app.conductor.build/users/api-keys"
        )
        action: RoutineAction = SchemaField(
            description="create a webhook-triggered routine, or rotate_webhook_url to "
            "replace an existing routine's webhook URL",
            default=RoutineAction.CREATE,
            advanced=False,
        )
        routine_id: str = SchemaField(
            description="Routine ID (rotate_webhook_url)", default="", advanced=False
        )
        name: str = SchemaField(
            description="Routine name (create)", default="", advanced=False
        )
        prompt: str = SchemaField(
            description="Prompt the agent runs each time the webhook fires (create)",
            default="",
            advanced=False,
        )
        project_id: str = SchemaField(
            description="Project (repository) the routine runs in (create). "
            "Find IDs with Get Account.",
            default="",
            advanced=False,
        )
        agent: ConductorAgent = SchemaField(
            description="Agent that runs the routine",
            default=ConductorAgent.CLAUDE,
        )
        model: str = SchemaField(
            description="Model id such as fable-5-1, opus-5-5-1m, sonnet-5-1m, "
            "gpt-6-astra or auto. Leave empty for Conductor's default.",
            default="",
        )
        effort: ConductorEffort = SchemaField(
            description="Reasoning effort; leave empty for the default",
            default=ConductorEffort.DEFAULT,
        )
        enabled: bool = SchemaField(
            description="Whether the routine is enabled on creation", default=True
        )

    class Output(BlockSchemaOutput):
        routine_id: str = SchemaField(description="ID of the routine")
        webhook_url: str = SchemaField(
            description="Webhook URL that triggers the routine. POST to it to run "
            "the routine; this is the only time the URL is shown."
        )
        routine: dict = SchemaField(description="Full routine object")

    def __init__(self):
        super().__init__(
            id="017de31f-f2ce-429d-8d71-a2447d76b544",
            description="Create a Conductor routine (a saved prompt that runs a fresh "
            "agent in a project whenever its webhook URL is called) or rotate a "
            "routine's webhook secret. Returns the webhook URL.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "action": RoutineAction.CREATE.value,
                    "name": "Nightly triage",
                    "prompt": "Triage new issues",
                    "project_id": "proj_1",
                },
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "action": RoutineAction.ROTATE_WEBHOOK_URL.value,
                    "routine_id": "rt_1",
                },
            ],
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("routine_id", "rt_1"),
                ("webhook_url", "https://api.conductor.build/hooks/abc"),
                ("routine", lambda r: r["id"] == "rt_1"),
                ("routine_id", "rt_1"),
                ("webhook_url", "https://api.conductor.build/hooks/abc"),
                ("routine", lambda r: r["id"] == "rt_1"),
            ],
            test_mock={
                "_perform": lambda *args, **kwargs: {
                    "id": "rt_1",
                    "name": "Nightly triage",
                    "prompt": "Triage new issues",
                    "repoUrl": "https://github.com/x/y",
                    "agent": "claude",
                    "enabled": True,
                    "triggers": [
                        {
                            "id": "tr_1",
                            "type": "webhook",
                            "enabled": True,
                            "webhookUrl": "https://api.conductor.build/hooks/abc",
                        }
                    ],
                }
            },
        )

    async def _perform(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        if input_data.action == RoutineAction.ROTATE_WEBHOOK_URL:
            return await client.rotate_routine_secret(input_data.routine_id)
        payload = clean(
            {
                "name": input_data.name,
                "prompt": input_data.prompt,
                "projectId": input_data.project_id,
                "agent": input_data.agent,
                "model": input_data.model,
                "effort": input_data.effort,
            }
        )
        payload["enabled"] = input_data.enabled
        payload["triggers"] = [{"type": "webhook"}]
        return await client.create_routine(payload)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.action == RoutineAction.CREATE:
            required = {
                "name": input_data.name,
                "prompt": input_data.prompt,
                "project_id": input_data.project_id,
            }
            missing = [key for key, value in required.items() if not value.strip()]
            if missing:
                raise BlockInputError(
                    message=f"{', '.join(missing)} required to create a routine",
                    block_name=self.name,
                    block_id=self.id,
                )
        elif not input_data.routine_id:
            raise BlockInputError(
                message="routine_id is required to rotate a routine secret",
                block_name=self.name,
                block_id=self.id,
            )

        try:
            routine = await self._perform(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Routine {input_data.action.value} failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "routine_id", str(routine.get("id") or input_data.routine_id)
        yield "webhook_url", webhook_url(routine)
        yield "routine", routine
