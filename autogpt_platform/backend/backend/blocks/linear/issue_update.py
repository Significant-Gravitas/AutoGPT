from pydantic import field_validator

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
from backend.util.exceptions import BlockExecutionError

from ._config import (
    TEST_CREDENTIALS_INPUT_OAUTH,
    TEST_CREDENTIALS_OAUTH,
    LinearScope,
    linear,
)
from ._issue_mutations import LinearIssueClient
from ._issue_updates import IssueChanges
from .models import Issue


class LinearUpdateIssueBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linear.credentials_field(
            required_scopes={LinearScope.WRITE}
        )
        issue_id: str = SchemaField(
            description="Issue UUID or identifier, such as ENG-123.", min_length=1
        )
        changes: IssueChanges = SchemaField(
            description="Fields to change. Omitted values leave existing fields unchanged; use clear_fields to clear nullable fields."
        )

        @field_validator("issue_id")
        @classmethod
        def validate_issue_id(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("Issue ID must not be blank")
            return value.strip()

    class Output(BlockSchemaOutput):
        issue: Issue = SchemaField(description="The updated Linear issue.")

    def __init__(self):
        super().__init__(
            id="29d22e4a-3386-4802-8e96-54558279d6bf",
            description="Updates a Linear issue's title, description, status, priority, assignee, labels, due date, or estimate.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
                "issue_id": "ENG-123",
                "changes": {"priority": 0},
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[("issue", Issue)],
            test_mock={
                "update_issue": lambda *args, **kwargs: Issue(
                    id="1108b684-5175-4f44-95b5-f1e9cd95f821",
                    identifier="ENG-123",
                    title="Example issue",
                    description=None,
                    priority=0,
                )
            },
        )

    @staticmethod
    async def update_issue(
        credentials: OAuth2Credentials | APIKeyCredentials,
        issue_id: str,
        changes: IssueChanges,
    ) -> Issue | None:
        return await LinearIssueClient(credentials=credentials).update_issue(
            issue_id, changes
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        issue = await self.update_issue(
            credentials, input_data.issue_id, input_data.changes
        )
        if issue is None:
            raise BlockExecutionError(
                "Linear did not confirm the issue update", self.name, self.id
            )
        yield "issue", issue
