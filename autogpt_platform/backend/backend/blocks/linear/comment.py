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
from .models import CreateCommentResponse


class LinearCreateCommentBlock(Block):
    """Block for creating comments on Linear issues"""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = linear.credentials_field(
            description="Linear credentials with comment creation permissions",
            required_scopes={"read", "comments:create"},
        )
        issue_id: String = SchemaField(description="ID of the issue to comment on")
        comment: String = SchemaField(description="Comment text to add to the issue")

    class Output(BlockSchema):
        comment_id: String = SchemaField(description="ID of the created comment")
        comment_body: String = SchemaField(
            description="Text content of the created comment"
        )
        error: String = SchemaField(
            description="Error message if comment creation failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="8f7d3a2e-9b5c-4c6a-8f1d-7c8b3e4a5d6c",
            description="Creates a new comment on a Linear issue",
            input_schema=LinearCreateCommentBlock.Input,
            output_schema=LinearCreateCommentBlock.Output,
            categories={BlockCategory.PRODUCTIVITY},
        )

    @staticmethod
    async def create_comment(
        credentials: OAuth2Credentials, issue_id: str, comment: str
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
        credentials: OAuth2Credentials,
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
