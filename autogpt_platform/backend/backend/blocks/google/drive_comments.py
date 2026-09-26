import asyncio
from typing import Any, Optional

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._drive import GoogleDriveFile, GoogleDriveFileField
from ._drive_api import (
    DRIVE_READONLY_SCOPE,
    build_drive_service,
    drive_error,
    require_file,
)

COMMENT_FIELDS = (
    "nextPageToken, comments(id, content, author(displayName, emailAddress), "
    "createdTime, modifiedTime, resolved, quotedFileContent(value), "
    "replies(id, content, action, deleted, author(displayName, emailAddress), "
    "createdTime))"
)


class DriveCommentReply(BaseModel):
    id: str = Field(description="Reply ID")
    author: str = Field(description="Name of the person who replied")
    author_email: Optional[str] = Field(
        default=None, description="Their email, when Google shares it"
    )
    content: str = Field(description="The reply's text")
    action: Optional[str] = Field(
        default=None,
        description="'resolve' or 'reopen' when the reply changed the thread's state",
    )
    created_time: Optional[str] = Field(
        default=None, description="When it was posted (RFC 3339)"
    )


class DriveComment(BaseModel):
    id: str = Field(description="Comment ID")
    author: str = Field(description="Name of the person who commented")
    author_email: Optional[str] = Field(
        default=None, description="Their email, when Google shares it"
    )
    content: str = Field(description="The comment's text")
    quoted_text: Optional[str] = Field(
        default=None, description="The text in the file the comment is attached to"
    )
    resolved: bool = Field(description="Whether the thread is resolved")
    created_time: Optional[str] = Field(
        default=None, description="When it was posted (RFC 3339)"
    )
    modified_time: Optional[str] = Field(
        default=None, description="When it was last changed (RFC 3339)"
    )
    replies: list[DriveCommentReply] = Field(
        default_factory=list, description="Replies in the thread, oldest first"
    )


_TEST_PICKED_FILE = {
    "id": "1a2b3c4d5e6f7g8h9i0j",
    "name": "Q3 Report",
    "mimeType": "application/vnd.google-apps.document",
}
_TEST_COMMENT_RESOURCE = {
    "id": "AAAA1",
    "content": "Can we double-check this number?",
    "author": {"displayName": "Sam Lee", "emailAddress": "sam@example.com"},
    "createdTime": "2026-09-20T15:30:00.000Z",
    "modifiedTime": "2026-09-21T09:00:00.000Z",
    "resolved": True,
    "quotedFileContent": {"value": "Revenue grew 12%"},
    "replies": [
        {
            "id": "BBBB1",
            "content": "Checked, it's right.",
            "action": "resolve",
            "author": {"displayName": "Alex Kim"},
            "createdTime": "2026-09-21T09:00:00.000Z",
        }
    ],
}


def to_drive_comment(item: dict[str, Any]) -> DriveComment:
    """Map a Drive API comment resource, dropping deleted replies."""
    return DriveComment(
        id=item["id"],
        author=(item.get("author") or {}).get("displayName", ""),
        author_email=(item.get("author") or {}).get("emailAddress"),
        content=item.get("content", ""),
        quoted_text=(item.get("quotedFileContent") or {}).get("value"),
        resolved=bool(item.get("resolved")),
        created_time=item.get("createdTime"),
        modified_time=item.get("modifiedTime"),
        replies=[
            DriveCommentReply(
                id=reply["id"],
                author=(reply.get("author") or {}).get("displayName", ""),
                author_email=(reply.get("author") or {}).get("emailAddress"),
                content=reply.get("content", ""),
                action=reply.get("action"),
                created_time=reply.get("createdTime"),
            )
            for reply in item.get("replies", [])
            if not reply.get("deleted")
        ],
    )


class GoogleDriveListCommentsBlock(Block):
    """List the comment threads on a Drive file, with their replies."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = GoogleDriveFileField(
            title="File",
            description="The Doc, Sheet, Slides deck or other Drive file whose comments to list",
            credentials_scopes=[DRIVE_READONLY_SCOPE],
        )
        include_resolved: bool = SchemaField(
            description="Include resolved threads", default=True
        )
        max_results: int = SchemaField(
            description="Maximum number of threads to return",
            default=100,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        comments: list[DriveComment] = SchemaField(
            description="Comment threads, oldest first, each with its replies"
        )
        comment: DriveComment = SchemaField(description="Each comment thread")

    def __init__(self):
        test_comment = to_drive_comment(_TEST_COMMENT_RESOURCE)
        super().__init__(
            id="9e245618-7641-4ec8-a706-fe7c37848d8b",
            description=(
                "List the comment threads on a Google Doc, Sheet, Slides deck or "
                "other Drive file: who said what, the text each comment is on, "
                "replies, and whether the thread is resolved."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveListCommentsBlock.Input,
            output_schema=GoogleDriveListCommentsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("comments", [test_comment]),
                ("comment", test_comment),
            ],
            test_mock={
                "_list_comments": lambda *args, **kwargs: [_TEST_COMMENT_RESOURCE]
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        service = build_drive_service(credentials)
        try:
            items = await asyncio.to_thread(
                self._list_comments,
                service,
                file.id,
                input_data.max_results,
                input_data.include_resolved,
            )
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        comments = [to_drive_comment(item) for item in items]
        yield "comments", comments
        for comment in comments:
            yield "comment", comment

    @staticmethod
    def _list_comments(
        service, file_id: str, max_results: int, include_resolved: bool
    ) -> list[dict]:
        comments: list[dict] = []
        page_token = None
        while len(comments) < max_results:
            response = (
                service.comments()
                .list(
                    fileId=file_id,
                    fields=COMMENT_FIELDS,
                    pageSize=100,
                    pageToken=page_token,
                )
                .execute()
            )
            comments.extend(
                item
                for item in response.get("comments", [])
                if include_resolved or not item.get("resolved")
            )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return comments[:max_results]
