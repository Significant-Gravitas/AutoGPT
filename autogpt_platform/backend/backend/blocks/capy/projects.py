"""Block that lists Capy projects, to resolve the project IDs threads need."""

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

from ._api import CapyClient
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, capy_credentials_field
from ._testdata import TEST_PROJECT
from ._types import Project


class CapyListProjectsBlock(Block):
    """List the Capy projects the key can see, with their repositories."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = capy_credentials_field()

    class Output(BlockSchemaOutput):
        projects: list[Project] = SchemaField(
            description="Every project visible to the key, with its repositories"
        )
        project: Project = SchemaField(description="Each project, one at a time")
        project_ids: list[str] = SchemaField(
            description="IDs of the projects, for the project_id input elsewhere"
        )

    def __init__(self):
        super().__init__(
            id="0f0db29e-9c01-44a1-9d4e-8659bdde6b52",
            description=(
                "Lists the Capy projects your API key can see, with the "
                "repositories each one covers. Use it to find the project ID "
                "a new Capy thread runs in."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=CapyListProjectsBlock.Input,
            output_schema=CapyListProjectsBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("projects", [TEST_PROJECT]),
                ("project", TEST_PROJECT),
                ("project_ids", [TEST_PROJECT.id]),
            ],
            test_mock={"list_projects": lambda *args, **kwargs: [TEST_PROJECT]},
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def list_projects(credentials: APIKeyCredentials) -> list[Project]:
        return await CapyClient(credentials).list_projects()

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        projects = await self.list_projects(credentials)
        yield "projects", projects
        for project in projects:
            yield "project", project
        yield "project_ids", [project.id for project in projects]
