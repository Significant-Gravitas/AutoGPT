from backend.blocks.linear._api import LinearAPIException, LinearClient
from backend.blocks.linear._auth import (
    LINEAR_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS_INPUT_OAUTH,
    TEST_CREDENTIALS_OAUTH,
    LinearCredentials,
    LinearCredentialsField,
    LinearCredentialsInput,
    LinearScope,
)
from backend.blocks.linear.models import Project
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class LinearSearchProjectsBlock(Block):
    """Block for searching projects on Linear"""

    class Input(BlockSchema):
        credentials: LinearCredentialsInput = LinearCredentialsField(
            scopes=[LinearScope.READ],
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
    def search_projects(
        credentials: LinearCredentials,
        term: str,
    ) -> list[Project]:
        client = LinearClient(credentials=credentials)
        response: list[Project] = client.try_search_projects(term=term)
        return response

    def run(
        self, input_data: Input, *, credentials: LinearCredentials, **kwargs
    ) -> BlockOutput:
        """Execute the project search"""
        try:
            projects = self.search_projects(
                credentials=credentials,
                term=input_data.term,
            )

            yield "projects", projects

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
