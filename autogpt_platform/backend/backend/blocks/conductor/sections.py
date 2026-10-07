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

from ._api import ConductorClient, SectionAction
from ._config import conductor


class ConductorManageSectionBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = conductor.credentials_field(
            description="Conductor API key from app.conductor.build/users/api-keys"
        )
        action: SectionAction = SchemaField(
            description="create a personal section, or delete one",
            default=SectionAction.CREATE,
            advanced=False,
        )
        name: str = SchemaField(
            description="Section name (create)", default="", advanced=False
        )
        emoji: str = SchemaField(
            description="Optional emoji shown next to the section name (create)",
            default="",
        )
        section_id: str = SchemaField(
            description="Section ID (delete)", default="", advanced=False
        )

    class Output(BlockSchemaOutput):
        section_id: str = SchemaField(
            description="ID of the created or deleted section"
        )
        section: dict = SchemaField(
            description="The section: id, name, emoji, workspaceIds"
        )
        removed_workspace_ids: list[str] = SchemaField(
            description="Workspaces that were in the section (delete only)"
        )

    def __init__(self):
        super().__init__(
            id="7f5d1f15-feec-45fa-95e5-a37f748cb678",
            description="Create or delete a Conductor cloud section. Sections group "
            "workspaces in the sidebar; move workspaces with Manage Workspace.",
            categories={BlockCategory.DEVELOPER_TOOLS},
            effect=BlockEffect.EXTERNAL,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "action": SectionAction.CREATE.value,
                    "name": "Bugs",
                    "emoji": "🐛",
                },
                {
                    "credentials": conductor.get_test_credentials().model_dump(),
                    "action": SectionAction.DELETE.value,
                    "section_id": "sec_1",
                },
            ],
            test_credentials=conductor.get_test_credentials(),
            test_output=[
                ("section_id", "sec_1"),
                ("section", {"id": "sec_1", "name": "Bugs", "workspaceIds": []}),
                ("removed_workspace_ids", []),
                ("section_id", "sec_1"),
                ("section", {"id": "sec_1", "name": "Bugs", "workspaceIds": ["ws_1"]}),
                ("removed_workspace_ids", ["ws_1"]),
            ],
            test_mock={
                "_perform": lambda credentials, input_data: {
                    "section": {
                        "id": "sec_1",
                        "name": "Bugs",
                        "workspaceIds": (
                            []
                            if input_data.action == SectionAction.CREATE
                            else ["ws_1"]
                        ),
                    }
                }
            },
        )

    async def _perform(
        self, credentials: APIKeyCredentials, input_data: Input
    ) -> dict[str, Any]:
        client = ConductorClient(credentials)
        if input_data.action == SectionAction.CREATE:
            return await client.create_section(input_data.name, input_data.emoji)
        return await client.delete_section(input_data.section_id)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.action == SectionAction.CREATE and not input_data.name.strip():
            raise BlockInputError(
                message="name is required to create a section",
                block_name=self.name,
                block_id=self.id,
            )
        if input_data.action == SectionAction.DELETE and not input_data.section_id:
            raise BlockInputError(
                message="section_id is required to delete a section",
                block_name=self.name,
                block_id=self.id,
            )

        try:
            result = await self._perform(credentials, input_data)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Section {input_data.action.value} failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        section = result.get("section") or {}
        yield "section_id", str(section.get("id") or input_data.section_id)
        yield "section", section
        yield "removed_workspace_ids", (
            list(section.get("workspaceIds") or [])
            if input_data.action == SectionAction.DELETE
            else []
        )
