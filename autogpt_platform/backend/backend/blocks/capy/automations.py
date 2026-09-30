"""Blocks that list, pause and delete Capy automations."""

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

from ._automations_api import CapyAutomationsClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._testdata import TEST_AUTOMATION, TEST_PROJECT
from ._types import Automation


def _automation_id_field() -> str:
    return SchemaField(
        description="The Capy automation ID (see Capy List Automations)",
        placeholder=TEST_AUTOMATION.id,
    )


class CapyListAutomationsBlock(Block):
    """List Capy automations with their triggers, state and run counts."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        project_id: str = SchemaField(
            description=(
                "Only automations in this project. Empty lists every project "
                "the key can see."
            ),
            default="",
        )
        limit: int = SchemaField(
            description="Maximum number of automations to return",
            default=20,
            ge=1,
            le=100,
        )
        cursor: str = SchemaField(
            description="Paging cursor from a previous call's next_cursor",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        automations: list[Automation] = SchemaField(
            description="The automations on this page"
        )
        automation: Automation = SchemaField(
            description="Each automation, one at a time"
        )
        next_cursor: str = SchemaField(
            description="Pass back as cursor for the next page; empty on the last"
        )

    def __init__(self):
        super().__init__(
            id="8d3da6b4-a878-4ec0-9ae6-b333c00dd050",
            description=(
                "Lists your Capy automations: what triggers each one, whether "
                "it is on, how many runs it has started and when it last fired."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyListAutomationsBlock.Input,
            output_schema=CapyListAutomationsBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "project_id": TEST_PROJECT.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("automations", [TEST_AUTOMATION]),
                ("automation", TEST_AUTOMATION),
                ("next_cursor", ""),
            ],
            test_mock={
                "list_automations": lambda *args, **kwargs: ([TEST_AUTOMATION], None)
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_automations(
        credentials: APIKeyCredentials, project_id: str, limit: int, cursor: str
    ) -> tuple[list[Automation], str | None]:
        return await CapyAutomationsClient(credentials).list_automations(
            project_id, limit, cursor
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        automations, cursor = await self.list_automations(
            credentials, input_data.project_id, input_data.limit, input_data.cursor
        )
        yield "automations", automations
        for automation in automations:
            yield "automation", automation
        yield "next_cursor", cursor or ""


class CapySetAutomationEnabledBlock(Block):
    """Turn a Capy automation on, or pause it."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        automation_id: str = _automation_id_field()
        enabled: bool = SchemaField(
            description="True turns the automation on; false, the default, pauses it",
            default=False,
        )

    class Output(BlockSchemaOutput):
        automation: Automation = SchemaField(description="The updated automation")
        enabled: bool = SchemaField(
            description="Whether it is now listening for its trigger"
        )

    def __init__(self):
        super().__init__(
            id="a0a5d683-5751-4e84-9612-d3d7aad1e547",
            description=(
                "Turns a Capy automation on, or pauses it so its trigger starts "
                "no runs until it is turned back on."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapySetAutomationEnabledBlock.Input,
            output_schema=CapySetAutomationEnabledBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "automation_id": TEST_AUTOMATION.id,
                "enabled": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("automation", TEST_AUTOMATION.model_copy(update={"enabled": False})),
                ("enabled", False),
            ],
            test_mock={
                "set_enabled": lambda *args, **kwargs: TEST_AUTOMATION.model_copy(
                    update={"enabled": False}
                )
            },
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def set_enabled(
        credentials: APIKeyCredentials, automation_id: str, enabled: bool
    ) -> Automation:
        return await CapyAutomationsClient(credentials).set_automation_enabled(
            automation_id, enabled
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        automation = await self.set_enabled(
            credentials, input_data.automation_id, input_data.enabled
        )
        yield "automation", automation
        yield "enabled", automation.enabled


class CapyDeleteAutomationBlock(Block):
    """Delete a Capy automation so it stops starting runs."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()
        automation_id: str = _automation_id_field()

    class Output(BlockSchemaOutput):
        automation: Automation = SchemaField(description="The deleted automation")
        deleted: bool = SchemaField(description="Whether Capy deleted it")

    def __init__(self):
        super().__init__(
            id="2ff5f020-9df0-48cb-8428-ef9c957a0e9b",
            description=(
                "Deletes a Capy automation so its trigger starts no more runs. "
                "Capy keeps deleted automations restorable."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyDeleteAutomationBlock.Input,
            output_schema=CapyDeleteAutomationBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "automation_id": TEST_AUTOMATION.id,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "automation",
                    TEST_AUTOMATION.model_copy(
                        update={"enabled": False, "deleted": True}
                    ),
                ),
                ("deleted", True),
            ],
            test_mock={
                "delete_automation": lambda *args, **kwargs: TEST_AUTOMATION.model_copy(
                    update={"enabled": False, "deleted": True}
                )
            },
            effect=BlockEffect.EXTERNAL,
        )

    @staticmethod
    async def delete_automation(
        credentials: APIKeyCredentials, automation_id: str
    ) -> Automation:
        return await CapyAutomationsClient(credentials).delete_automation(automation_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        automation = await self.delete_automation(credentials, input_data.automation_id)
        yield "automation", automation
        yield "deleted", automation.deleted
