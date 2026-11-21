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


class IssueLifecycleInput(BlockSchemaInput):
    credentials: CredentialsMetaInput = linear.credentials_field(
        required_scopes={LinearScope.WRITE}
    )
    issue_id: str = SchemaField(
        description="Issue UUID or identifier, such as ENG-123.", min_length=1
    )

    @field_validator("issue_id")
    @classmethod
    def validate_issue_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Issue ID must not be blank")
        return value.strip()


class IssueLifecycleOutput(BlockSchemaOutput):
    issue_id: str = SchemaField(description="ID or identifier of the affected issue.")
    success: bool = SchemaField(description="Whether Linear confirmed the operation.")


class LinearArchiveIssueBlock(Block):
    Input = IssueLifecycleInput
    Output = IssueLifecycleOutput

    def __init__(self):
        super().__init__(
            id="9dc6fdde-74f1-4ba0-be88-2f9cce7c5b5d",
            description="Archives a Linear issue, removing it from active views while retaining it in the archive.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
                "issue_id": "ENG-123",
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[("issue_id", "ENG-123"), ("success", True)],
            test_mock={"archive_issue": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def archive_issue(
        credentials: OAuth2Credentials | APIKeyCredentials, issue_id: str
    ) -> bool:
        return await LinearIssueClient(credentials=credentials).archive_issue(issue_id)

    async def run(
        self,
        input_data: IssueLifecycleInput,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        if not await self.archive_issue(credentials, input_data.issue_id):
            raise BlockExecutionError(
                "Linear did not confirm the issue archive", self.name, self.id
            )
        yield "issue_id", input_data.issue_id
        yield "success", True


class LinearDeleteIssueBlock(Block):
    Input = IssueLifecycleInput
    Output = IssueLifecycleOutput

    def __init__(self):
        super().__init__(
            id="f8fe8378-f09e-4e94-a3da-00002c48ba95",
            description="Deletes a Linear issue using Linear's recoverable trash behavior. Does not request immediate permanent deletion.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            input_schema=self.Input,
            output_schema=self.Output,
            is_sensitive_action=True,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
                "issue_id": "ENG-123",
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[("issue_id", "ENG-123"), ("success", True)],
            test_mock={"delete_issue": lambda *args, **kwargs: True},
        )

    @staticmethod
    async def delete_issue(
        credentials: OAuth2Credentials | APIKeyCredentials, issue_id: str
    ) -> bool:
        return await LinearIssueClient(credentials=credentials).delete_issue(issue_id)

    async def run(
        self,
        input_data: IssueLifecycleInput,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs
    ) -> BlockOutput:
        if not await self.delete_issue(credentials, input_data.issue_id):
            raise BlockExecutionError(
                "Linear did not confirm the issue deletion", self.name, self.id
            )
        yield "issue_id", input_data.issue_id
        yield "success", True
