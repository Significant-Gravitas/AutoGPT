from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
)

from ._api import LinearAPIException, LinearClient
from ._config import (
    TEST_CREDENTIALS_INPUT_OAUTH,
    TEST_CREDENTIALS_OAUTH,
    LinearScope,
    linear,
)
from .models import CreateIssueResponse, Issue


class LinearCreateIssueBlock(Block):
    """Block for creating issues on Linear"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with issue creation permissions",
            required_scopes={LinearScope.ISSUES_CREATE},
        )
        title: str = SchemaField(description="Title of the issue")
        description: str | None = SchemaField(description="Description of the issue")
        team_name: str = SchemaField(
            description="Name of the team to create the issue on"
        )
        priority: int | None = SchemaField(
            description="Priority of the issue",
            default=None,
            ge=0,
            le=4,
        )
        project_name: str | None = SchemaField(
            description="Name of the project to create the issue on",
            default=None,
        )

    class Output(BlockSchemaOutput):
        issue_id: str = SchemaField(description="ID of the created issue")
        issue_title: str = SchemaField(description="Title of the created issue")

    def __init__(self):
        super().__init__(
            id="f9c68f55-dcca-40a8-8771-abf9601680aa",
            description="Creates a new issue on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            test_input={
                "title": "Test issue",
                "description": "Test description",
                "team_name": "Test team",
                "project_name": "Test project",
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[("issue_id", "abc123"), ("issue_title", "Test issue")],
            test_mock={
                "create_issue": lambda *args, **kwargs: (
                    "abc123",
                    "Test issue",
                )
            },
        )

    @staticmethod
    async def create_issue(
        credentials: OAuth2Credentials | APIKeyCredentials,
        team_name: str,
        title: str,
        description: str | None = None,
        priority: int | None = None,
        project_name: str | None = None,
    ) -> tuple[str, str]:
        client = LinearClient(credentials=credentials)
        team_id = await client.try_get_team_by_name(team_name=team_name)
        project_id: str | None = None
        if project_name:
            projects = await client.try_search_projects(term=project_name)
            if projects:
                project_id = projects[0].id
            else:
                raise LinearAPIException("Project not found", status_code=404)
        response: CreateIssueResponse = await client.try_create_issue(
            team_id=team_id,
            title=title,
            description=description,
            priority=priority,
            project_id=project_id,
        )
        return response.issue.identifier, response.issue.title

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the issue creation"""
        try:
            issue_id, issue_title = await self.create_issue(
                credentials=credentials,
                team_name=input_data.team_name,
                title=input_data.title,
                description=input_data.description,
                priority=input_data.priority,
                project_name=input_data.project_name,
            )

            yield "issue_id", issue_id
            yield "issue_title", issue_title

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"


class LinearSearchIssuesBlock(Block):
    """Block for searching issues on Linear"""

    class Input(BlockSchemaInput):
        term: str = SchemaField(description="Term to search for issues")
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with read permissions",
            required_scopes={LinearScope.READ},
        )

    class Output(BlockSchemaOutput):
        issues: list[Issue] = SchemaField(description="List of issues")

    def __init__(self):
        super().__init__(
            id="b5a2a0e6-26b4-4c5b-8a42-bc79e9cb65c2",
            description="Searches for issues on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "term": "Test issue",
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[
                (
                    "issues",
                    [
                        Issue(
                            id="abc123",
                            identifier="abc123",
                            title="Test issue",
                            description="Test description",
                            priority=1,
                        )
                    ],
                )
            ],
            test_mock={
                "search_issues": lambda *args, **kwargs: [
                    Issue(
                        id="abc123",
                        identifier="abc123",
                        title="Test issue",
                        description="Test description",
                        priority=1,
                    )
                ]
            },
        )

    @staticmethod
    async def search_issues(
        credentials: OAuth2Credentials | APIKeyCredentials,
        term: str,
    ) -> list[Issue]:
        client = LinearClient(credentials=credentials)
        response: list[Issue] = await client.try_search_issues(term=term)
        return response

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the issue search"""
        try:
            issues = await self.search_issues(
                credentials=credentials, term=input_data.term
            )
            yield "issues", issues
        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"


class LinearGetProjectIssuesBlock(Block):
    """Block for getting issues from a Linear project filtered by status and assignee"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with read permissions",
            required_scopes={LinearScope.READ},
        )
        project: str = SchemaField(description="Name of the project to get issues from")
        status: str = SchemaField(
            description="Status/state name to filter issues by (e.g., 'In Progress', 'Done')"
        )
        is_assigned: bool = SchemaField(
            description="Filter by assignee status - True to get assigned issues, False to get unassigned issues",
            default=False,
        )
        include_comments: bool = SchemaField(
            description="Whether to include comments in the response",
            default=False,
        )

    class Output(BlockSchemaOutput):
        issues: list[Issue] = SchemaField(
            description="List of issues matching the criteria"
        )

    def __init__(self):
        super().__init__(
            id="c7d3f1e8-45a9-4b2c-9f81-3e6a8d7c5b1a",
            description="Gets issues from a Linear project filtered by status and assignee",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            test_input={
                "project": "Test Project",
                "status": "In Progress",
                "is_assigned": False,
                "include_comments": False,
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[
                (
                    "issues",
                    [
                        Issue(
                            id="abc123",
                            identifier="TST-123",
                            title="Test issue",
                            description="Test description",
                            priority=1,
                        )
                    ],
                ),
            ],
            test_mock={
                "get_project_issues": lambda *args, **kwargs: [
                    Issue(
                        id="abc123",
                        identifier="TST-123",
                        title="Test issue",
                        description="Test description",
                        priority=1,
                    )
                ]
            },
        )

    @staticmethod
    async def get_project_issues(
        credentials: OAuth2Credentials | APIKeyCredentials,
        project: str,
        status: str,
        is_assigned: bool,
        include_comments: bool,
    ) -> list[Issue]:
        client = LinearClient(credentials=credentials)
        response: list[Issue] = await client.try_get_issues(
            project=project,
            status=status,
            is_assigned=is_assigned,
            include_comments=include_comments,
        )
        return response

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute getting project issues"""
        issues = await self.get_project_issues(
            credentials=credentials,
            project=input_data.project,
            status=input_data.status,
            is_assigned=input_data.is_assigned,
            include_comments=input_data.include_comments,
        )
        yield "issues", issues
