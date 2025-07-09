from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)

from ._api import LinearAPIException, LinearClient
from ._config import (
    LINEAR_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS_INPUT_OAUTH,
    TEST_CREDENTIALS_OAUTH,
    LinearScope,
    linear,
)
from .models import Project


class LinearSearchProjectsBlock(Block):
    """Block for searching projects on Linear"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with read permissions",
            required_scopes={LinearScope.READ},
        )
        term: str = SchemaField(description="Term to search for projects")

    class Output(BlockSchema):
        projects: list[Project] = SchemaField(description="List of projects")
        error: str = SchemaField(description="Error message if issue creation failed")

    def __init__(self):
        super().__init__(
            id="446a1d35-9d8f-4ac5-83ea-7684ec50e6af",
            description="Searches for projects on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            test_input={
                "term": "Test project",
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
            },
            disabled=not LINEAR_OAUTH_IS_CONFIGURED,
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[
                (
                    "projects",
                    [
                        Project(
                            id="abc123",
                            name="Test project",
                            description="Test description",
                            priority=1,
                            progress=1,
                            content="Test content",
                        )
                    ],
                )
            ],
            test_mock={
                "search_projects": lambda *args, **kwargs: [
                    Project(
                        id="abc123",
                        name="Test project",
                        description="Test description",
                        priority=1,
                        progress=1,
                        content="Test content",
                    )
                ]
            },
        )

    @staticmethod
    async def search_projects(
        credentials: OAuth2Credentials | APIKeyCredentials,
        term: str,
    ) -> list[Project]:
        client = LinearClient(credentials=credentials)
        response: list[Project] = await client.try_search_projects(term=term)
        return response

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the project search"""
        try:
            projects = await self.search_projects(
                credentials=credentials,
                term=input_data.term,
            )

            yield "projects", projects

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
