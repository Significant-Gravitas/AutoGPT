from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    OAuth2Credentials,
    SchemaField,
    String,
)

from ._api import LinearAPIException, LinearClient
from ._config import linear
from .models import CreateIssueResponse, Issue


class LinearCreateIssueBlock(Block):
    """Block for creating issues on Linear"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with issue creation permissions",
            required_scopes={"read", "issues:create"},
        )
        title: String = SchemaField(description="Title of the issue")
        description: String = SchemaField(
            description="Description of the issue", default=""
        )
        team_name: String = SchemaField(
            description="Name of the team to create the issue on"
        )
        priority: int = SchemaField(
            description="Priority of the issue (0-4, where 0 is no priority, 1 is urgent, 2 is high, 3 is normal, 4 is low)",
            default=3,
            ge=0,
            le=4,
        )
        project_name: String = SchemaField(
            description="Name of the project to create the issue on",
            default="",
        )

    class Output(BlockSchema):
        issue_id: String = SchemaField(description="ID of the created issue")
        issue_title: String = SchemaField(description="Title of the created issue")
        error: String = SchemaField(
            description="Error message if issue creation failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="f9c68f55-dcca-40a8-8771-abf9601680aa",
            description="Creates a new issue on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY},
        )

    @staticmethod
    async def create_issue(
        credentials: OAuth2Credentials,
        team_name: str,
        title: str,
        description: str = "",
        priority: int = 3,
        project_name: str = "",
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
            description=description if description else None,
            priority=priority if priority != 3 else None,
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

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with read permissions",
            required_scopes={"read"},
        )
        term: String = SchemaField(description="Term to search for issues")

    class Output(BlockSchema):
        issues: list[Issue] = SchemaField(description="List of issues")
        error: String = SchemaField(
            description="Error message if search failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="b5a2a0e6-26b4-4c5b-8a42-bc79e9cb65c2",
            description="Searches for issues on Linear",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY},
        )

    @staticmethod
    async def search_issues(
        credentials: OAuth2Credentials,
        term: str,
    ) -> list[Issue]:
        client = LinearClient(credentials=credentials)
        response: list[Issue] = await client.try_search_issues(term=term)
        return response

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials,
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
