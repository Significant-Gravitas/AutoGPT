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
from .models import CreateCommentResponse


class LinearCreateCommentBlock(Block):
    """Block for creating comments on Linear issues"""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with comment creation permissions",
            required_scopes={LinearScope.COMMENTS_CREATE},
        )
        issue_id: str = SchemaField(description="ID of the issue to comment on")
        comment: str = SchemaField(description="Comment text to add to the issue")

    class Output(BlockSchemaOutput):
        comment_id: str = SchemaField(description="ID of the created comment")
        comment_body: str = SchemaField(
            description="Text content of the created comment"
        )

    def __init__(self):
        super().__init__(
            id="8f7d3a2e-9b5c-4c6a-8f1d-7c8b3e4a5d6c",
            description="Creates a new comment on a Linear issue",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            test_input={
                "issue_id": "TEST-123",
                "comment": "Test comment",
                "credentials": TEST_CREDENTIALS_INPUT_OAUTH,
            },
            test_credentials=TEST_CREDENTIALS_OAUTH,
            test_output=[("comment_id", "abc123"), ("comment_body", "Test comment")],
            test_mock={
                "create_comment": lambda *args, **kwargs: (
                    "abc123",
                    "Test comment",
                )
            },
        )

    @staticmethod
    async def create_comment(
        credentials: OAuth2Credentials | APIKeyCredentials, issue_id: str, comment: str
    ) -> tuple[str, str]:
        client = LinearClient(credentials=credentials)
        response: CreateCommentResponse = await client.try_create_comment(
            issue_id=issue_id, comment=comment
        )
        return response.comment.id, response.comment.body

    async def run(
        self,
        input_data: Input,
        *,
        credentials: OAuth2Credentials | APIKeyCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the comment creation"""
        try:
            comment_id, comment_body = await self.create_comment(
                credentials=credentials,
                issue_id=input_data.issue_id,
                comment=input_data.comment,
            )

            yield "comment_id", comment_id
            yield "comment_body", comment_body

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
