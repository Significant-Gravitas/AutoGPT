import asyncio
from typing import Any

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._batch_update import batch_update_error
from ._drive import GoogleDriveFile, GoogleDriveFileField
from .docs import _build_docs_service, _make_document_output, _validate_document_file

_TEST_DOCUMENT = {
    "id": "1abc123def456",
    "name": "Proposal",
    "mimeType": "application/vnd.google-apps.document",
}
_TEST_REQUESTS = [
    {
        "replaceAllText": {
            "containsText": {"text": "{{client}}", "matchCase": True},
            "replaceText": "Acme Corp",
        }
    }
]


class GoogleDocsBatchUpdateBlock(Block):
    """Apply a raw list of Google Docs API requests to a document."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="The Google Doc to update",
            allowed_views=["DOCUMENTS"],
        )
        requests: list[dict[str, Any]] = SchemaField(
            description=(
                "Google Docs API batchUpdate requests, applied in order and all "
                'or nothing, e.g. [{"insertText": {"location": {"index": 1}, '
                '"text": "Hello\\n"}}]. Request types: '
                "https://developers.google.com/workspace/docs/api/reference/rest/v1/documents/request"
            ),
        )
        required_revision_id: str = SchemaField(
            description=(
                "Only apply the update if the document is still at this revision "
                "(from an earlier read or update). Empty always applies it."
            ),
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        replies: list[dict[str, Any]] = SchemaField(
            description="One reply per request, in order (empty for requests that return nothing)"
        )
        revision_id: str = SchemaField(
            description="The document's revision after the update"
        )
        document: GoogleDriveFile = SchemaField(
            description="The document, for chaining"
        )

    def __init__(self):
        super().__init__(
            id="fd41c1ed-a18c-49f8-9da2-090724452d1b",
            description=(
                "Apply any Google Docs API batchUpdate requests to a document in one "
                "all-or-nothing call: named ranges, bullets, headers, footnotes, "
                "images and anything the other Google Docs blocks don't cover."
            ),
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsBatchUpdateBlock.Input,
            output_schema=GoogleDocsBatchUpdateBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"document": _TEST_DOCUMENT, "requests": _TEST_REQUESTS},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("replies", [{"replaceAllText": {"occurrencesChanged": 3}}]),
                ("revision_id", "ALm37BVTest"),
                (
                    "document",
                    _make_document_output(
                        GoogleDriveFile.model_validate(_TEST_DOCUMENT)
                    ),
                ),
            ],
            test_mock={
                "_batch_update": lambda *args, **kwargs: {
                    "replies": [{"replaceAllText": {"occurrencesChanged": 3}}],
                    "writeControl": {"requiredRevisionId": "ALm37BVTest"},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        document = input_data.document
        problem = (
            _validate_document_file(document) if document else "Pick a Google Doc."
        )
        if problem:
            raise BlockInputError(
                message=problem, block_name=self.name, block_id=self.id
            )
        if not input_data.requests:
            raise BlockInputError(
                message="Add at least one request.",
                block_name=self.name,
                block_id=self.id,
            )
        body: dict[str, Any] = {"requests": input_data.requests}
        if input_data.required_revision_id:
            body["writeControl"] = {
                "requiredRevisionId": input_data.required_revision_id
            }
        service = _build_docs_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._batch_update, service, document.id, body
            )
        except HttpError as e:
            raise batch_update_error(e, "Docs", self.name, self.id) from e
        yield "replies", result.get("replies", [])
        yield "revision_id", (result.get("writeControl") or {}).get(
            "requiredRevisionId", ""
        )
        yield "document", _make_document_output(document)

    @staticmethod
    def _batch_update(service, document_id: str, body: dict) -> dict:
        return (
            service.documents().batchUpdate(documentId=document_id, body=body).execute()
        )
