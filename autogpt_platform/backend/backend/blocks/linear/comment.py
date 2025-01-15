from typing import Optional

from backend.blocks.linear.models import CreateCommentResponse, CreateCommentResponseWrapper
from backend.data.block import Block, BlockCategory, BlockSchema, BlockOutput
from backend.data.model import CredentialsField, SchemaField
from backend.blocks.linear._auth import (
    LinearCredentials,
    LinearCredentialsField,
    LinearCredentialsInput,
    LinearScope,
)
from backend.blocks.linear._api import LinearClient, LinearAPIException


class CreateCommentBlock(Block):
    """Block for creating comments on Linear issues"""

    class Input(BlockSchema):
        credentials: LinearCredentialsInput = LinearCredentialsField(
            scopes=[LinearScope.COMMENTS_CREATE],
        )
        issue_id: str = SchemaField(description="ID of the issue to comment on")
        comment: str = SchemaField(description="Comment text to add to the issue")

    class Output(BlockSchema):
        comment_id: str = SchemaField(description="ID of the created comment")
        comment_body: str = SchemaField(
            description="Text content of the created comment"
        )
        error: str = SchemaField(description="Error message if comment creation failed")

    def __init__(self):
        super().__init__(
            id="8f7d3a2e-9b5c-4c6a-8f1d-7c8b3e4a5d6c",  # Generated UUID
            description="Creates a new comment on a Linear issue",
            input_schema=self.Input,
            output_schema=self.Output,
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.ISSUE_TRACKING},
            test_input={
                "issue_id": "TEST-123",
                "comment": "Test comment",
                "credentials": {"api_key": "test_key"},
            },
            test_output=[("comment_id", "abc123"), ("comment_body", "Test comment")],
        )

    def run(
        self, input_data: Input, *, credentials: LinearCredentials, **kwargs
    ) -> BlockOutput:
        """Execute the comment creation"""
        try:
            client = LinearClient(credentials=credentials)

            response: CreateCommentResponse = client.try_create_comment(
                issue_id=input_data.issue_id, comment=input_data.comment
            )

            if response.success:
                comment = response.comment
                yield "comment_id", comment.id
                yield "comment_body", comment.body
            else:
                yield "error", "Failed to create comment"

        except LinearAPIException as e:
            yield "error", str(e)
        except Exception as e:
            yield "error", f"Unexpected error: {str(e)}"
