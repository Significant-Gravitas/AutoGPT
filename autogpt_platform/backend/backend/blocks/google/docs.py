import asyncio
import re
from enum import Enum
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from gravitas_md2gdocs import to_requests

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.google._drive import GoogleDriveFile, GoogleDriveFileField
from backend.data.model import SchemaField
from backend.util.settings import Settings

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)

settings = Settings()
GOOGLE_DOCS_DISABLED = not GOOGLE_OAUTH_IS_CONFIGURED


# ============ Enums ============


class PublicAccessRole(str, Enum):
    READER = "reader"
    COMMENTER = "commenter"


class ShareRole(str, Enum):
    READER = "reader"
    WRITER = "writer"
    COMMENTER = "commenter"


# ============ Helper Functions ============


def _build_docs_service(credentials: GoogleCredentials):
    """Build Google Docs API service."""
    creds = Credentials(
        token=(
            credentials.access_token.get_secret_value()
            if credentials.access_token
            else None
        ),
        refresh_token=(
            credentials.refresh_token.get_secret_value()
            if credentials.refresh_token
            else None
        ),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.secrets.google_client_id,
        client_secret=settings.secrets.google_client_secret,
        scopes=credentials.scopes,
    )
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def _build_drive_service(credentials: GoogleCredentials):
    """Build Google Drive API service for file operations."""
    creds = Credentials(
        token=(
            credentials.access_token.get_secret_value()
            if credentials.access_token
            else None
        ),
        refresh_token=(
            credentials.refresh_token.get_secret_value()
            if credentials.refresh_token
            else None
        ),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.secrets.google_client_id,
        client_secret=settings.secrets.google_client_secret,
        scopes=credentials.scopes,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _validate_document_file(file: GoogleDriveFile) -> str | None:
    """Validate that a file is a Google Doc."""
    if not file.id:
        return "No document ID provided"
    if file.mime_type and file.mime_type != "application/vnd.google-apps.document":
        return f"File is not a Google Doc (type: {file.mime_type})"
    return None


def _parse_hex_color_to_rgb_floats(value: str) -> tuple[float, float, float] | None:
    """
    Parse a CSS-like hex color string into normalized RGB floats.

    Supports:
    - #RGB / RGB (shorthand)
    - #RRGGBB / RRGGBB

    Returns None for malformed inputs.
    """
    if not value:
        return None

    raw = value.strip()
    if raw.startswith("#"):
        raw = raw[1:]

    if not re.fullmatch(r"[0-9a-fA-F]{3}([0-9a-fA-F]{3})?", raw):
        return None

    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)

    r = int(raw[0:2], 16) / 255.0
    g = int(raw[2:4], 16) / 255.0
    b = int(raw[4:6], 16) / 255.0
    return (r, g, b)


def _get_document_end_index(service, document_id: str) -> int:
    """Get the index at the end of the document body."""
    doc = service.documents().get(documentId=document_id).execute()
    body = doc.get("body", {})
    content = body.get("content", [])
    if content:
        last_element = content[-1]
        return last_element.get("endIndex", 1) - 1
    return 1


def _extract_text_from_content(content: list[dict]) -> str:
    """Extract plain text from document content structure."""
    text_parts = []
    for element in content:
        if "paragraph" in element:
            for elem in element["paragraph"].get("elements", []):
                if "textRun" in elem:
                    text_parts.append(elem["textRun"].get("content", ""))
        elif "table" in element:
            for row in element["table"].get("tableRows", []):
                for cell in row.get("tableCells", []):
                    cell_content = cell.get("content", [])
                    text_parts.append(_extract_text_from_content(cell_content))
    return "".join(text_parts)


# ============ Document Output Helper ============


def _make_document_output(file: GoogleDriveFile) -> GoogleDriveFile:
    """Create standardized document output for chaining."""
    return GoogleDriveFile(
        id=file.id,
        name=file.name,
        mimeType="application/vnd.google-apps.document",
        url=f"https://docs.google.com/document/d/{file.id}/edit",
        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
        isFolder=False,
        _credentials_id=file.credentials_id,
    )


# ============ Blocks ============


class GoogleDocsReadBlock(Block):
    """Read content from a Google Doc."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to read",
            allowed_views=["DOCUMENTS"],
        )

    class Output(BlockSchemaOutput):
        text: str = SchemaField(description="Plain text content of the document")
        title: str = SchemaField(description="Document title")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if read failed")

    def __init__(self):
        super().__init__(
            id="420a2b3c-5db2-4bda-82bc-a68a862a3d55",
            description="Read text content from a Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsReadBlock.Input,
            output_schema=GoogleDocsReadBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("text", "Hello World\nThis is a test document.\n"),
                ("title", "Test Document"),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_read_document": lambda *args, **kwargs: {
                    "text": "Hello World\nThis is a test document.\n",
                    "title": "Test Document",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._read_document,
                service,
                input_data.document.id,
            )
            yield "text", result["text"]
            yield "title", result["title"]
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to read document: {str(e)}"

    def _read_document(self, service, document_id: str) -> dict:
        doc = service.documents().get(documentId=document_id).execute()
        title = doc.get("title", "")
        body = doc.get("body", {})
        content = body.get("content", [])
        text = _extract_text_from_content(content)
        return {"text": text, "title": title}


class GoogleDocsCreateBlock(Block):
    """Create a new Google Doc."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/drive.file"]
        )
        title: str = SchemaField(description="Title for the new document")
        initial_content: str = SchemaField(
            default="",
            description="Optional initial text content",
        )

    class Output(BlockSchemaOutput):
        document: GoogleDriveFile = SchemaField(description="The created document")
        document_id: str = SchemaField(description="ID of the created document")
        document_url: str = SchemaField(description="URL to open the document")
        error: str = SchemaField(description="Error message if creation failed")

    def __init__(self):
        super().__init__(
            id="d430d941-cf81-4f84-8b19-2e3f670b2fca",
            description="Create a new Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsCreateBlock.Input,
            output_schema=GoogleDocsCreateBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "title": "My New Document",
                "initial_content": "Hello, this is the initial content.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "document",
                    GoogleDriveFile(
                        id="new_doc_123",
                        name="My New Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/new_doc_123/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=TEST_CREDENTIALS_INPUT["id"],
                    ),
                ),
                ("document_id", "new_doc_123"),
                ("document_url", "https://docs.google.com/document/d/new_doc_123/edit"),
            ],
            test_mock={
                "_create_document": lambda *args, **kwargs: {
                    "document_id": "new_doc_123",
                    "document_url": "https://docs.google.com/document/d/new_doc_123/edit",
                    "title": "My New Document",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.title:
            yield "error", "Document title is required"
            return

        try:
            drive_service = _build_drive_service(credentials)
            docs_service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._create_document,
                drive_service,
                docs_service,
                input_data.title,
                input_data.initial_content,
            )
            doc_id = result["document_id"]
            doc_url = result["document_url"]
            yield "document", GoogleDriveFile(
                id=doc_id,
                name=input_data.title,
                mimeType="application/vnd.google-apps.document",
                url=doc_url,
                iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                isFolder=False,
                _credentials_id=input_data.credentials.id,
            )
            yield "document_id", doc_id
            yield "document_url", doc_url
        except Exception as e:
            yield "error", f"Failed to create document: {str(e)}"

    def _create_document(
        self, drive_service, docs_service, title: str, initial_content: str
    ) -> dict:
        # Create the document
        file_metadata = {
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
        }
        result = drive_service.files().create(body=file_metadata).execute()
        document_id = result.get("id")
        document_url = f"https://docs.google.com/document/d/{document_id}/edit"

        # Add initial content if provided
        if initial_content:
            requests = [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": initial_content,
                    }
                }
            ]
            docs_service.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

        return {
            "document_id": document_id,
            "document_url": document_url,
            "title": title,
        }


class GoogleDocsAppendPlainTextBlock(Block):
    """Append plain text to the end of a Google Doc (no formatting)."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to append to",
            allowed_views=["DOCUMENTS"],
        )
        text: str = SchemaField(
            description="Plain text to append (no formatting applied)"
        )
        add_newline: bool = SchemaField(
            default=True,
            description="Add a newline before the appended text",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the append operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if append failed")

    def __init__(self):
        super().__init__(
            id="ddc29d9f-78dc-4682-8787-c8a76f00cf38",
            description="Append plain text to the end of a Google Doc (no formatting applied)",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsAppendPlainTextBlock.Input,
            output_schema=GoogleDocsAppendPlainTextBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "text": "This is appended text.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "characters_added": 23}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_append_text": lambda *args, **kwargs: {
                    "success": True,
                    "characters_added": 23,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._append_text,
                service,
                input_data.document.id,
                input_data.text,
                input_data.add_newline,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to append text: {str(e)}"

    def _append_text(
        self, service, document_id: str, text: str, add_newline: bool
    ) -> dict:
        end_index = _get_document_end_index(service, document_id)
        text_to_insert = ("\n" if add_newline else "") + text

        requests = [
            {
                "insertText": {
                    "location": {"index": end_index},
                    "text": text_to_insert,
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        return {"success": True, "characters_added": len(text_to_insert)}


class GoogleDocsInsertPlainTextBlock(Block):
    """Insert plain text at a specific position in a Google Doc (no formatting)."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to insert into",
            allowed_views=["DOCUMENTS"],
        )
        text: str = SchemaField(
            description="Plain text to insert (no formatting applied)"
        )
        index: int = SchemaField(
            default=1,
            description="Position index to insert at (1 = start of document)",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the insert operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if insert failed")

    def __init__(self):
        super().__init__(
            id="0443fdbc-ebb0-49a4-a3ea-6ace9c14da22",
            description="Insert plain text at a specific position in a Google Doc (no formatting applied)",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsInsertPlainTextBlock.Input,
            output_schema=GoogleDocsInsertPlainTextBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "text": "Inserted text here. ",
                "index": 1,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "characters_inserted": 20}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_insert_text": lambda *args, **kwargs: {
                    "success": True,
                    "characters_inserted": 20,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._insert_text,
                service,
                input_data.document.id,
                input_data.text,
                input_data.index,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to insert text: {str(e)}"

    def _insert_text(self, service, document_id: str, text: str, index: int) -> dict:
        requests = [
            {
                "insertText": {
                    "location": {"index": max(1, index)},
                    "text": text,
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        return {"success": True, "characters_inserted": len(text)}


class GoogleDocsFindReplacePlainTextBlock(Block):
    """Find and replace plain text in a Google Doc (no formatting applied to replacement)."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        find_text: str = SchemaField(description="Plain text to find")
        replace_text: str = SchemaField(
            description="Plain text to replace with (no formatting applied)"
        )
        match_case: bool = SchemaField(
            default=False,
            description="Match case when finding text",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result with replacement count")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="e5046ee2-b094-418e-a25e-c0f90c91721c",
            description="Find and replace plain text in a Google Doc (no formatting applied to replacement)",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsFindReplacePlainTextBlock.Input,
            output_schema=GoogleDocsFindReplacePlainTextBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "find_text": "old text",
                "replace_text": "new text",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "replacements_made": 3}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_find_replace": lambda *args, **kwargs: {
                    "success": True,
                    "replacements_made": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._find_replace,
                service,
                input_data.document.id,
                input_data.find_text,
                input_data.replace_text,
                input_data.match_case,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to find/replace: {str(e)}"

    def _find_replace(
        self,
        service,
        document_id: str,
        find_text: str,
        replace_text: str,
        match_case: bool,
    ) -> dict:
        requests = [
            {
                "replaceAllText": {
                    "containsText": {
                        "text": find_text,
                        "matchCase": match_case,
                    },
                    "replaceText": replace_text,
                }
            }
        ]

        response = (
            service.documents()
            .batchUpdate(documentId=document_id, body={"requests": requests})
            .execute()
        )

        # Get replacement count from response
        replies = response.get("replies", [])
        replacements = 0
        if replies and "replaceAllText" in replies[0]:
            replacements = replies[0]["replaceAllText"].get("occurrencesChanged", 0)

        return {"success": True, "replacements_made": replacements}


class GoogleDocsGetMetadataBlock(Block):
    """Get metadata about a Google Doc."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )

    class Output(BlockSchemaOutput):
        title: str = SchemaField(description="Document title")
        document_id: str = SchemaField(description="Document ID")
        revision_id: str = SchemaField(description="Current revision ID")
        document_url: str = SchemaField(description="URL to open the document")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="100bc806-acbf-4dc5-a3a2-998026b96516",
            description="Get metadata about a Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsGetMetadataBlock.Input,
            output_schema=GoogleDocsGetMetadataBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("title", "Test Document"),
                ("document_id", "1abc123def456"),
                ("revision_id", "rev_123"),
                (
                    "document_url",
                    "https://docs.google.com/document/d/1abc123def456/edit",
                ),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_metadata": lambda *args, **kwargs: {
                    "title": "Test Document",
                    "revision_id": "rev_123",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._get_metadata,
                service,
                input_data.document.id,
            )
            yield "title", result["title"]
            yield "document_id", input_data.document.id
            yield "revision_id", result["revision_id"]
            yield "document_url", f"https://docs.google.com/document/d/{input_data.document.id}/edit"
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to get metadata: {str(e)}"

    def _get_metadata(self, service, document_id: str) -> dict:
        doc = service.documents().get(documentId=document_id).execute()
        return {
            "title": doc.get("title", ""),
            "revision_id": doc.get("revisionId", ""),
        }


class GoogleDocsInsertTableBlock(Block):
    """Insert a table into a Google Doc, optionally with content."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        rows: int = SchemaField(
            default=3,
            description="Number of rows (ignored if content provided)",
        )
        columns: int = SchemaField(
            default=3,
            description="Number of columns (ignored if content provided)",
        )
        content: list[list[str]] = SchemaField(
            default=[],
            description="Optional 2D array of cell content, e.g. [['Header1', 'Header2'], ['Row1Col1', 'Row1Col2']]. If provided, rows/columns are derived from this.",
        )
        index: int = SchemaField(
            default=0,
            description="Position to insert table (0 = end of document)",
        )
        format_as_markdown: bool = SchemaField(
            default=False,
            description="Format cell content as Markdown (headers, bold, links, etc.)",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of table insertion")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="e104b3ab-dfef-45f9-9702-14e950988f53",
            description="Insert a table into a Google Doc, optionally with content and Markdown formatting",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsInsertTableBlock.Input,
            output_schema=GoogleDocsInsertTableBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "content": [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {
                        "success": True,
                        "rows": 2,
                        "columns": 2,
                        "cells_populated": 4,
                        "cells_found": 4,
                    },
                ),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_insert_table": lambda *args, **kwargs: {
                    "success": True,
                    "rows": 2,
                    "columns": 2,
                    "cells_populated": 4,
                    "cells_found": 4,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        # Determine rows/columns from content if provided
        content = input_data.content

        # Check if content is valid:
        # 1. Has at least one row with at least one cell (even if empty string)
        # 2. Has at least one non-empty cell value
        has_valid_structure = bool(content and any(len(row) > 0 for row in content))
        has_content = has_valid_structure and any(
            cell for row in content for cell in row
        )

        if has_content:
            # Use content dimensions - filter out empty rows for row count,
            # use max column count across all rows
            rows = len(content)
            columns = max(len(row) for row in content)
        else:
            # No valid content - use explicit rows/columns, clear content
            rows = input_data.rows
            columns = input_data.columns
            content = []  # Clear so we skip population step

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._insert_table,
                service,
                input_data.document.id,
                rows,
                columns,
                input_data.index,
                content,
                input_data.format_as_markdown,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to insert table: {str(e)}"

    def _insert_table(
        self,
        service,
        document_id: str,
        rows: int,
        columns: int,
        index: int,
        content: list[list[str]],
        format_as_markdown: bool,
    ) -> dict:
        # If index is 0, insert at end of document
        if index == 0:
            index = _get_document_end_index(service, document_id)

        # Insert the empty table structure
        requests = [
            {
                "insertTable": {
                    "rows": rows,
                    "columns": columns,
                    "location": {"index": index},
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        # If no content provided, we're done
        if not content:
            return {"success": True, "rows": rows, "columns": columns}

        # Fetch the document to find cell indexes
        doc = service.documents().get(documentId=document_id).execute()
        body_content = doc.get("body", {}).get("content", [])

        # Find all tables and pick the one we just inserted
        # (the one with highest startIndex that's >= our insert point, or the last one if inserted at end)
        tables_found = []
        for element in body_content:
            if "table" in element:
                tables_found.append(element)

        if not tables_found:
            return {
                "success": True,
                "rows": rows,
                "columns": columns,
                "warning": "Table created but could not find it to populate",
            }

        # If we inserted at end (index was high), take the last table
        # Otherwise, take the first table at or after our insert index
        table_element = None
        # Heuristic: rows * columns * 2 estimates the minimum index space a table
        # occupies (each cell has at least a start index and structural overhead).
        # This helps determine if our insert point was near the document end.
        estimated_table_size = rows * columns * 2
        if (
            index
            >= _get_document_end_index(service, document_id) - estimated_table_size
        ):
            # Likely inserted at end - use last table
            table_element = tables_found[-1]
        else:
            for tbl in tables_found:
                if tbl.get("startIndex", 0) >= index:
                    table_element = tbl
                    break
            if not table_element:
                table_element = tables_found[-1]

        # Extract cell start indexes from the table structure
        # Structure: table -> tableRows -> tableCells -> content[0] -> startIndex
        cell_positions: list[tuple[int, int, int]] = []  # (row, col, start_index)
        table_data = table_element.get("table", {})
        table_rows_list = table_data.get("tableRows", [])

        for row_idx, table_row in enumerate(table_rows_list):
            cells = table_row.get("tableCells", [])
            for col_idx, cell in enumerate(cells):
                cell_content = cell.get("content", [])
                if cell_content:
                    # Get the start index of the first element in the cell
                    first_element = cell_content[0]
                    cell_start = first_element.get("startIndex")
                    if cell_start is not None:
                        cell_positions.append((row_idx, col_idx, cell_start))

        if not cell_positions:
            return {
                "success": True,
                "rows": rows,
                "columns": columns,
                "warning": f"Table created but could not extract cell positions. Table has {len(table_rows_list)} rows.",
            }

        # Sort by index descending so we can insert in reverse order
        # (inserting later content first preserves earlier indexes)
        cell_positions.sort(key=lambda x: x[2], reverse=True)

        cells_populated = 0

        if format_as_markdown:
            # Markdown formatting: process each cell individually since
            # gravitas-md2gdocs requests may have complex interdependencies
            for row_idx, col_idx, cell_start in cell_positions:
                if row_idx < len(content) and col_idx < len(content[row_idx]):
                    cell_text = content[row_idx][col_idx]
                    if not cell_text:
                        continue
                    md_requests = to_requests(cell_text, start_index=cell_start)
                    if md_requests:
                        service.documents().batchUpdate(
                            documentId=document_id, body={"requests": md_requests}
                        ).execute()
                        cells_populated += 1
        else:
            # Plain text: batch all insertions into a single API call
            # Cells are sorted by index descending, so earlier requests
            # don't affect indices of later ones
            all_text_requests = []
            for row_idx, col_idx, cell_start in cell_positions:
                if row_idx < len(content) and col_idx < len(content[row_idx]):
                    cell_text = content[row_idx][col_idx]
                    if not cell_text:
                        continue
                    all_text_requests.append(
                        {
                            "insertText": {
                                "location": {"index": cell_start},
                                "text": cell_text,
                            }
                        }
                    )
                    cells_populated += 1

            if all_text_requests:
                service.documents().batchUpdate(
                    documentId=document_id, body={"requests": all_text_requests}
                ).execute()

        return {
            "success": True,
            "rows": rows,
            "columns": columns,
            "cells_populated": cells_populated,
            "cells_found": len(cell_positions),
        }


class GoogleDocsInsertPageBreakBlock(Block):
    """Insert a page break into a Google Doc."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        index: int = SchemaField(
            default=0,
            description="Position to insert page break (0 = end of document)",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of page break insertion")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="f199e674-803b-4ee8-8bbf-172e6512190b",
            description="Insert a page break into a Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsInsertPageBreakBlock.Input,
            output_schema=GoogleDocsInsertPageBreakBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_insert_page_break": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._insert_page_break,
                service,
                input_data.document.id,
                input_data.index,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to insert page break: {str(e)}"

    def _insert_page_break(self, service, document_id: str, index: int) -> dict:
        if index == 0:
            index = _get_document_end_index(service, document_id)

        requests = [
            {
                "insertPageBreak": {
                    "location": {"index": index},
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        return {"success": True}


class GoogleDocsDeleteContentBlock(Block):
    """Delete a range of content from a Google Doc."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        start_index: int = SchemaField(
            description="Start index of content to delete (must be >= 1, as index 0 is a section break)",
            ge=1,
        )
        end_index: int = SchemaField(description="End index of content to delete")

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of delete operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="5f9f9fa4-9071-4028-97c2-9d15fb422dc5",
            description="Delete a range of content from a Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsDeleteContentBlock.Input,
            output_schema=GoogleDocsDeleteContentBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "start_index": 10,
                "end_index": 50,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "characters_deleted": 40}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_delete_content": lambda *args, **kwargs: {
                    "success": True,
                    "characters_deleted": 40,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if input_data.start_index >= input_data.end_index:
            yield "error", "Start index must be less than end index"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._delete_content,
                service,
                input_data.document.id,
                input_data.start_index,
                input_data.end_index,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to delete content: {str(e)}"

    def _delete_content(
        self, service, document_id: str, start_index: int, end_index: int
    ) -> dict:
        requests = [
            {
                "deleteContentRange": {
                    "range": {
                        "startIndex": start_index,
                        "endIndex": end_index,
                    }
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        return {"success": True, "characters_deleted": end_index - start_index}


class ExportFormat(str, Enum):
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ODT = "application/vnd.oasis.opendocument.text"
    TXT = "text/plain"
    HTML = "text/html"
    EPUB = "application/epub+zip"
    RTF = "application/rtf"


class GoogleDocsExportBlock(Block):
    """Export a Google Doc to various formats."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to export",
            allowed_views=["DOCUMENTS"],
        )
        format: ExportFormat = SchemaField(
            default=ExportFormat.PDF,
            description="Export format",
        )

    class Output(BlockSchemaOutput):
        content: str = SchemaField(
            description="Exported content (base64 encoded for binary formats)"
        )
        mime_type: str = SchemaField(description="MIME type of exported content")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if export failed")

    def __init__(self):
        super().__init__(
            id="e32d5642-7b51-458c-bd83-75ff96fec299",
            description="Export a Google Doc to PDF, Word, text, or other formats",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsExportBlock.Input,
            output_schema=GoogleDocsExportBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "format": ExportFormat.TXT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("content", "This is the document content as plain text."),
                ("mime_type", "text/plain"),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_export_document": lambda *args, **kwargs: {
                    "content": "This is the document content as plain text.",
                    "mime_type": "text/plain",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            drive_service = _build_drive_service(credentials)
            result = await asyncio.to_thread(
                self._export_document,
                drive_service,
                input_data.document.id,
                input_data.format.value,
            )
            yield "content", result["content"]
            yield "mime_type", result["mime_type"]
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to export document: {str(e)}"

    def _export_document(self, service, document_id: str, mime_type: str) -> dict:
        import base64

        response = (
            service.files().export(fileId=document_id, mimeType=mime_type).execute()
        )

        # For text formats, return as string; for binary, base64 encode
        if mime_type in ["text/plain", "text/html"]:
            content = (
                response.decode("utf-8") if isinstance(response, bytes) else response
            )
        else:
            content = base64.b64encode(response).decode("utf-8")

        return {"content": content, "mime_type": mime_type}


class GoogleDocsFormatTextBlock(Block):
    """Apply formatting to text in a Google Doc."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        start_index: int = SchemaField(
            description="Start index of text to format (must be >= 1, as index 0 is a section break)",
            ge=1,
        )
        end_index: int = SchemaField(description="End index of text to format")
        bold: bool = SchemaField(
            default=False,
            description="Make text bold",
        )
        italic: bool = SchemaField(
            default=False,
            description="Make text italic",
        )
        underline: bool = SchemaField(
            default=False,
            description="Underline text",
        )
        font_size: int = SchemaField(
            default=0,
            description="Font size in points (0 = no change)",
        )
        foreground_color: str = SchemaField(
            default="",
            description="Text color as hex (e.g., #FF0000 for red)",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of format operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="04c38a7e-7ee5-4e1a-86c1-9727123577bc",
            description="Apply formatting (bold, italic, color, etc.) to text in a Google Doc",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsFormatTextBlock.Input,
            output_schema=GoogleDocsFormatTextBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "start_index": 2,
                "end_index": 10,
                "bold": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_format_text": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if input_data.start_index >= input_data.end_index:
            yield "error", "Start index must be less than end index"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._format_text,
                service,
                input_data.document.id,
                input_data.start_index,
                input_data.end_index,
                input_data.bold,
                input_data.italic,
                input_data.underline,
                input_data.font_size,
                input_data.foreground_color,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to format text: {str(e)}"

    def _format_text(
        self,
        service,
        document_id: str,
        start_index: int,
        end_index: int,
        bold: bool,
        italic: bool,
        underline: bool,
        font_size: int,
        foreground_color: str,
    ) -> dict:
        text_style: dict[str, Any] = {}
        fields = []

        if bold:
            text_style["bold"] = True
            fields.append("bold")
        if italic:
            text_style["italic"] = True
            fields.append("italic")
        if underline:
            text_style["underline"] = True
            fields.append("underline")
        if font_size > 0:
            text_style["fontSize"] = {"magnitude": font_size, "unit": "PT"}
            fields.append("fontSize")
        if foreground_color:
            rgb = _parse_hex_color_to_rgb_floats(foreground_color)
            if rgb is None:
                if not fields:
                    return {
                        "success": False,
                        "message": (
                            f"Invalid foreground_color: {foreground_color!r}. "
                            "Expected hex like #RGB or #RRGGBB."
                        ),
                    }
                # Ignore invalid color, but still apply other formatting.
                # This avoids failing the whole operation due to a single bad value.
                warning = (
                    f"Ignored invalid foreground_color: {foreground_color!r}. "
                    "Expected hex like #RGB or #RRGGBB."
                )
            else:
                r, g, b = rgb
                text_style["foregroundColor"] = {
                    "color": {"rgbColor": {"red": r, "green": g, "blue": b}}
                }
                fields.append("foregroundColor")
                warning = None
        else:
            warning = None

        if not fields:
            return {"success": True, "message": "No formatting options specified"}

        requests = [
            {
                "updateTextStyle": {
                    "range": {"startIndex": start_index, "endIndex": end_index},
                    "textStyle": text_style,
                    "fields": ",".join(fields),
                }
            }
        ]

        service.documents().batchUpdate(
            documentId=document_id, body={"requests": requests}
        ).execute()

        if warning:
            return {"success": True, "warning": warning}
        return {"success": True}


class GoogleDocsShareBlock(Block):
    """Share a Google Doc with specific users."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to share",
            allowed_views=["DOCUMENTS"],
        )
        email: str = SchemaField(
            default="",
            description="Email address to share with. Leave empty for link sharing.",
        )
        role: ShareRole = SchemaField(
            default=ShareRole.READER,
            description="Permission role for the user",
        )
        send_notification: bool = SchemaField(
            default=True,
            description="Send notification email to the user",
        )
        message: str = SchemaField(
            default="",
            description="Optional message to include in notification email",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the share operation")
        share_link: str = SchemaField(description="Link to the document")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if share failed")

    def __init__(self):
        super().__init__(
            id="4e7ec771-4cc8-4eb7-ae3d-46377ecdb5d2",
            description="Share a Google Doc with specific users",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsShareBlock.Input,
            output_schema=GoogleDocsShareBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "email": "test@example.com",
                "role": "reader",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("share_link", "https://docs.google.com/document/d/1abc123def456/edit"),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_share_document": lambda *args, **kwargs: {
                    "success": True,
                    "share_link": "https://docs.google.com/document/d/1abc123def456/edit",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_drive_service(credentials)
            result = await asyncio.to_thread(
                self._share_document,
                service,
                input_data.document.id,
                input_data.email,
                input_data.role,
                input_data.send_notification,
                input_data.message,
            )
            yield "result", {"success": True}
            yield "share_link", result["share_link"]
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to share document: {str(e)}"

    def _share_document(
        self,
        service,
        document_id: str,
        email: str,
        role: ShareRole,
        send_notification: bool,
        message: str,
    ) -> dict:
        share_link = f"https://docs.google.com/document/d/{document_id}/edit"

        if email:
            # Share with specific user
            permission = {"type": "user", "role": role.value, "emailAddress": email}

            kwargs: dict[str, Any] = {
                "fileId": document_id,
                "body": permission,
                "sendNotificationEmail": send_notification,
            }
            if message:
                kwargs["emailMessage"] = message

            service.permissions().create(**kwargs).execute()
        else:
            # Create "anyone with the link" permission for link sharing
            permission = {"type": "anyone", "role": role.value}
            service.permissions().create(
                fileId=document_id,
                body=permission,
            ).execute()

        return {"success": True, "share_link": share_link}


class GoogleDocsSetPublicAccessBlock(Block):
    """Make a Google Doc publicly accessible or private."""

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        public: bool = SchemaField(
            default=True,
            description="True to make public, False to make private",
        )
        role: PublicAccessRole = SchemaField(
            default=PublicAccessRole.READER,
            description="Permission role for public access",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the operation")
        share_link: str = SchemaField(description="Link to the document")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="d104f6e1-80af-4fe9-b5a1-3cab20081b6c",
            description="Make a Google Doc public or private",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsSetPublicAccessBlock.Input,
            output_schema=GoogleDocsSetPublicAccessBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "public": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "is_public": True}),
                (
                    "share_link",
                    "https://docs.google.com/document/d/1abc123def456/edit?usp=sharing",
                ),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_set_public_access": lambda *args, **kwargs: {
                    "success": True,
                    "is_public": True,
                    "share_link": "https://docs.google.com/document/d/1abc123def456/edit?usp=sharing",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_drive_service(credentials)
            result = await asyncio.to_thread(
                self._set_public_access,
                service,
                input_data.document.id,
                input_data.public,
                input_data.role,
            )
            yield "result", {"success": True, "is_public": result["is_public"]}
            yield "share_link", result["share_link"]
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to set public access: {str(e)}"

    def _set_public_access(
        self, service, document_id: str, public: bool, role: PublicAccessRole
    ) -> dict:
        share_link = f"https://docs.google.com/document/d/{document_id}/edit"

        if public:
            permission = {"type": "anyone", "role": role.value}
            service.permissions().create(fileId=document_id, body=permission).execute()
            share_link += "?usp=sharing"
        else:
            permissions = service.permissions().list(fileId=document_id).execute()
            for perm in permissions.get("permissions", []):
                if perm.get("type") == "anyone":
                    service.permissions().delete(
                        fileId=document_id, permissionId=perm["id"]
                    ).execute()

        return {"success": True, "is_public": public, "share_link": share_link}


# ============ Markdown Blocks ============


class GoogleDocsAppendMarkdownBlock(Block):
    """Append Markdown content to the end of a Google Doc.

    Converts Markdown to Google Docs formatting, supporting:
    - Headers (H1-H6)
    - Bold, italic, strikethrough
    - Inline code and code blocks
    - Links
    - Bulleted and numbered lists
    - Blockquotes

    Perfect for AI agents that generate Markdown output.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to append to",
            allowed_views=["DOCUMENTS"],
        )
        markdown: str = SchemaField(
            description="Markdown content to append to the document"
        )
        add_newline: bool = SchemaField(
            default=True,
            description="Add a newline before the appended content",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the append operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="60854b69-ecbd-4188-bd89-f7966a4d3b38",
            description="Append Markdown content to the end of a Google Doc with full formatting - ideal for LLM/AI output",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsAppendMarkdownBlock.Input,
            output_schema=GoogleDocsAppendMarkdownBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "markdown": "# Hello World\n\nThis is **bold** and *italic* text.\n\n- Item 1\n- Item 2",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "requests_count": 5}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_append_markdown": lambda *args, **kwargs: {
                    "success": True,
                    "requests_count": 5,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.markdown:
            yield "error", "No markdown content provided"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._append_markdown,
                service,
                input_data.document.id,
                input_data.markdown,
                input_data.add_newline,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to append markdown: {str(e)}"

    def _append_markdown(
        self,
        service,
        document_id: str,
        markdown: str,
        add_newline: bool,
    ) -> dict:
        end_index = _get_document_end_index(service, document_id)

        # Optionally add a newline before the content
        if add_newline and end_index > 1:
            newline_requests = [
                {"insertText": {"location": {"index": end_index}, "text": "\n"}}
            ]
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": newline_requests}
            ).execute()
            end_index += 1

        # Convert markdown to Google Docs requests
        requests = to_requests(markdown, start_index=end_index)

        if requests:
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

        return {"success": True, "requests_count": len(requests)}


class GoogleDocsReplaceAllWithMarkdownBlock(Block):
    """Replace entire Google Doc content with Markdown.

    Clears the document and inserts formatted Markdown content.
    Supports headers, bold, italic, lists, links, code blocks, etc.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to replace content in",
            allowed_views=["DOCUMENTS"],
        )
        markdown: str = SchemaField(
            description="Markdown content to replace the document with"
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the replace operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="b6cfb2de-5f0b-437c-b29d-45aebbda9c00",
            description="Replace entire Google Doc content with formatted Markdown - ideal for LLM/AI output",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsReplaceAllWithMarkdownBlock.Input,
            output_schema=GoogleDocsReplaceAllWithMarkdownBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "markdown": "# New Document\n\nThis replaces everything.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "requests_count": 3}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_replace_all_with_markdown": lambda *args, **kwargs: {
                    "success": True,
                    "requests_count": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.markdown:
            yield "error", "No markdown content provided"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._replace_all_with_markdown,
                service,
                input_data.document.id,
                input_data.markdown,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to replace document with markdown: {str(e)}"

    def _replace_all_with_markdown(
        self,
        service,
        document_id: str,
        markdown: str,
    ) -> dict:
        # Delete all existing content
        doc_end = _get_document_end_index(service, document_id)
        if doc_end > 1:
            delete_requests = [
                {
                    "deleteContentRange": {
                        "range": {"startIndex": 1, "endIndex": doc_end}
                    }
                }
            ]
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": delete_requests}
            ).execute()

        # Insert markdown at beginning
        requests = to_requests(markdown, start_index=1)

        if requests:
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

        return {"success": True, "requests_count": len(requests)}


class GoogleDocsInsertMarkdownAtBlock(Block):
    """Insert Markdown content at a specific position in a Google Doc.

    Converts Markdown to Google Docs formatting and inserts at the specified index.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to insert into",
            allowed_views=["DOCUMENTS"],
        )
        markdown: str = SchemaField(description="Markdown content to insert")
        index: int = SchemaField(
            default=1,
            description="Position index to insert at (1 = start of document)",
            ge=1,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the insert operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="76e94b04-e02f-4981-8cb8-47ece1be18b4",
            description="Insert formatted Markdown at a specific position in a Google Doc - ideal for LLM/AI output",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsInsertMarkdownAtBlock.Input,
            output_schema=GoogleDocsInsertMarkdownAtBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "markdown": "## Inserted Section\n\nThis was inserted.",
                "index": 1,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "requests_count": 3}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_insert_markdown_at": lambda *args, **kwargs: {
                    "success": True,
                    "requests_count": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.markdown:
            yield "error", "No markdown content provided"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._insert_markdown_at,
                service,
                input_data.document.id,
                input_data.markdown,
                input_data.index,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to insert markdown: {str(e)}"

    def _insert_markdown_at(
        self,
        service,
        document_id: str,
        markdown: str,
        index: int,
    ) -> dict:
        requests = to_requests(markdown, start_index=index)

        if requests:
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

        return {"success": True, "requests_count": len(requests)}


class GoogleDocsReplaceRangeWithMarkdownBlock(Block):
    """Replace a specific range (by index) in a Google Doc with Markdown.

    Deletes content between start and end indices, then inserts formatted Markdown.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        markdown: str = SchemaField(
            description="Markdown content to insert in place of the range"
        )
        start_index: int = SchemaField(
            description="Start index of the range to replace (must be >= 1)",
            ge=1,
        )
        end_index: int = SchemaField(
            description="End index of the range to replace",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the replace operation")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="9e43a905-a918-4da0-8874-dfddd3c46953",
            description="Replace a specific index range in a Google Doc with formatted Markdown - ideal for LLM/AI output",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsReplaceRangeWithMarkdownBlock.Input,
            output_schema=GoogleDocsReplaceRangeWithMarkdownBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "markdown": "**Replaced content**",
                "start_index": 10,
                "end_index": 50,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"success": True, "requests_count": 2, "characters_deleted": 40},
                ),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_replace_range_with_markdown": lambda *args, **kwargs: {
                    "success": True,
                    "requests_count": 2,
                    "characters_deleted": 40,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.markdown:
            yield "error", "No markdown content provided"
            return

        if input_data.start_index >= input_data.end_index:
            yield "error", "Start index must be less than end index"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._replace_range_with_markdown,
                service,
                input_data.document.id,
                input_data.markdown,
                input_data.start_index,
                input_data.end_index,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to replace range with markdown: {str(e)}"

    def _replace_range_with_markdown(
        self,
        service,
        document_id: str,
        markdown: str,
        start_index: int,
        end_index: int,
    ) -> dict:
        # Delete the range first
        delete_requests = [
            {
                "deleteContentRange": {
                    "range": {"startIndex": start_index, "endIndex": end_index}
                }
            }
        ]
        service.documents().batchUpdate(
            documentId=document_id, body={"requests": delete_requests}
        ).execute()

        # Insert markdown at the start of the deleted range
        requests = to_requests(markdown, start_index=start_index)

        if requests:
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": requests}
            ).execute()

        return {
            "success": True,
            "requests_count": len(requests),
            "characters_deleted": end_index - start_index,
        }


class GoogleDocsReplaceContentWithMarkdownBlock(Block):
    """Find text in a Google Doc and replace it with formatted Markdown.

    Perfect for template workflows - use placeholders like {{INTRO}} or {{SUMMARY}}
    and replace them with formatted Markdown content.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc",
            allowed_views=["DOCUMENTS"],
        )
        find_text: str = SchemaField(
            description="Text to find and replace (e.g., '{{PLACEHOLDER}}' or any text)"
        )
        markdown: str = SchemaField(
            description="Markdown content to replace the found text with"
        )
        match_case: bool = SchemaField(
            default=False,
            description="Match case when finding text",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result with replacement count")
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="2cc58467-90a9-4ef8-a7a7-700784f93b76",
            description="Find text and replace it with formatted Markdown - ideal for LLM/AI output and templates",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsReplaceContentWithMarkdownBlock.Input,
            output_schema=GoogleDocsReplaceContentWithMarkdownBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "find_text": "{{PLACEHOLDER}}",
                "markdown": "# Replaced Header\n\nThis is the **replacement** content.",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"success": True, "replacements_made": 1, "requests_count": 4},
                ),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_replace_content_with_markdown": lambda *args, **kwargs: {
                    "success": True,
                    "replacements_made": 1,
                    "requests_count": 4,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.find_text:
            yield "error", "No find text provided"
            return

        if not input_data.markdown:
            yield "error", "No markdown content provided"
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._replace_content_with_markdown,
                service,
                input_data.document.id,
                input_data.find_text,
                input_data.markdown,
                input_data.match_case,
            )
            yield "result", result
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to replace content with markdown: {str(e)}"

    def _find_text_positions(
        self, service, document_id: str, find_text: str, match_case: bool
    ) -> list[tuple[int, int]]:
        """Find all positions of the search text using actual document indices.

        Iterates through document content and uses the real startIndex/endIndex
        from text runs, rather than trying to map plain text offsets to indices.
        """
        doc = service.documents().get(documentId=document_id).execute()
        body = doc.get("body", {})
        content = body.get("content", [])

        positions = []
        search_text = find_text if match_case else find_text.lower()

        def search_in_content(elements: list[dict]) -> None:
            """Recursively search through content elements."""
            for element in elements:
                if "paragraph" in element:
                    for text_elem in element["paragraph"].get("elements", []):
                        if "textRun" in text_elem:
                            text_run = text_elem["textRun"]
                            text_content = text_run.get("content", "")
                            start_index = text_elem.get("startIndex", 0)

                            # Search within this text run
                            text_to_search = (
                                text_content if match_case else text_content.lower()
                            )
                            offset = 0
                            while True:
                                pos = text_to_search.find(search_text, offset)
                                if pos == -1:
                                    break
                                # Calculate actual document indices
                                doc_start = start_index + pos
                                doc_end = doc_start + len(find_text)
                                positions.append((doc_start, doc_end))
                                offset = pos + 1

                elif "table" in element:
                    # Search within table cells
                    for row in element["table"].get("tableRows", []):
                        for cell in row.get("tableCells", []):
                            search_in_content(cell.get("content", []))

        search_in_content(content)
        return positions

    def _replace_content_with_markdown(
        self,
        service,
        document_id: str,
        find_text: str,
        markdown: str,
        match_case: bool,
    ) -> dict:
        # Find all positions of the text
        positions = self._find_text_positions(
            service, document_id, find_text, match_case
        )

        if not positions:
            return {"success": True, "replacements_made": 0, "requests_count": 0}

        total_requests = 0
        replacements_made = 0

        # Process in reverse order to maintain correct indices
        for start_index, end_index in reversed(positions):
            # Build combined request: delete first, then insert markdown
            # Combining into single batchUpdate reduces API calls by half
            combined_requests = [
                {
                    "deleteContentRange": {
                        "range": {"startIndex": start_index, "endIndex": end_index}
                    }
                }
            ]

            # Get markdown insert requests
            md_requests = to_requests(markdown, start_index=start_index)
            if md_requests:
                combined_requests.extend(md_requests)

            # Execute delete + insert in single API call
            service.documents().batchUpdate(
                documentId=document_id, body={"requests": combined_requests}
            ).execute()

            total_requests += len(combined_requests)
            replacements_made += 1

        return {
            "success": True,
            "replacements_made": replacements_made,
            "requests_count": total_requests,
        }


class GoogleDocsGetStructureBlock(Block):
    """Get document structure with index positions for precise editing operations.

    Returns content segments with their start/end indexes, making it easy to
    target specific parts of the document for insertion, deletion, or formatting.
    """

    class Input(BlockSchemaInput):
        document: GoogleDriveFile = GoogleDriveFileField(
            title="Document",
            description="Select a Google Doc to analyze",
            allowed_views=["DOCUMENTS"],
        )
        detailed: bool = SchemaField(
            default=False,
            description="Return full hierarchical structure instead of flat segments",
        )

    class Output(BlockSchemaOutput):
        segments: list[dict] = SchemaField(
            description="Flat list of content segments with indexes (when detailed=False)"
        )
        structure: dict = SchemaField(
            description="Full hierarchical document structure (when detailed=True)"
        )
        document: GoogleDriveFile = SchemaField(description="The document for chaining")
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="e0561cc1-2154-4abf-bd06-79509348a18e",
            description="Get document structure with index positions for precise editing operations",
            categories={BlockCategory.DATA},
            input_schema=GoogleDocsGetStructureBlock.Input,
            output_schema=GoogleDocsGetStructureBlock.Output,
            disabled=GOOGLE_DOCS_DISABLED,
            test_input={
                "document": {
                    "id": "1abc123def456",
                    "name": "Test Document",
                    "mimeType": "application/vnd.google-apps.document",
                },
                "detailed": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "segments",
                    [
                        {
                            "type": "paragraph",
                            "text": "Hello World",
                            "start_index": 1,
                            "end_index": 12,
                        },
                        {
                            "type": "paragraph",
                            "text": "Second paragraph",
                            "start_index": 13,
                            "end_index": 29,
                        },
                    ],
                ),
                ("structure", {}),
                (
                    "document",
                    GoogleDriveFile(
                        id="1abc123def456",
                        name="Test Document",
                        mimeType="application/vnd.google-apps.document",
                        url="https://docs.google.com/document/d/1abc123def456/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/docs_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_structure": lambda *args, **kwargs: {
                    "segments": [
                        {
                            "type": "paragraph",
                            "text": "Hello World",
                            "start_index": 1,
                            "end_index": 12,
                        },
                        {
                            "type": "paragraph",
                            "text": "Second paragraph",
                            "start_index": 13,
                            "end_index": 29,
                        },
                    ],
                    "structure": {},
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.document:
            yield "error", "No document selected"
            return

        validation_error = _validate_document_file(input_data.document)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_docs_service(credentials)
            result = await asyncio.to_thread(
                self._get_structure,
                service,
                input_data.document.id,
                input_data.detailed,
            )
            yield "segments", result["segments"]
            yield "structure", result["structure"]
            yield "document", _make_document_output(input_data.document)
        except Exception as e:
            yield "error", f"Failed to get document structure: {str(e)}"

    def _extract_paragraph_text(self, paragraph: dict) -> str:
        """Extract plain text from a paragraph element."""
        text_parts = []
        for elem in paragraph.get("elements", []):
            if "textRun" in elem:
                text_parts.append(elem["textRun"].get("content", ""))
        return "".join(text_parts).rstrip("\n")

    def _get_paragraph_style(self, paragraph: dict) -> dict:
        """Get paragraph style information."""
        style = paragraph.get("paragraphStyle", {})
        named_style = style.get("namedStyleType", "NORMAL_TEXT")

        # Map named styles to heading levels
        heading_map = {
            "HEADING_1": 1,
            "HEADING_2": 2,
            "HEADING_3": 3,
            "HEADING_4": 4,
            "HEADING_5": 5,
            "HEADING_6": 6,
        }

        if named_style in heading_map:
            return {"heading_level": heading_map[named_style]}
        return {}

    def _process_table_detailed(self, table_element: dict) -> dict:
        """Process table for detailed hierarchical output."""
        table = table_element.get("table", {})
        table_rows = table.get("tableRows", [])

        rows_data = []
        for table_row in table_rows:
            cells_data = []
            for cell in table_row.get("tableCells", []):
                cell_content = cell.get("content", [])
                cell_text = ""
                cell_start = None
                cell_end = None

                for content_elem in cell_content:
                    if "paragraph" in content_elem:
                        cell_text += self._extract_paragraph_text(
                            content_elem["paragraph"]
                        )
                        if cell_start is None:
                            cell_start = content_elem.get("startIndex")
                        cell_end = content_elem.get("endIndex")

                cells_data.append(
                    {
                        "text": cell_text,
                        "start_index": cell_start,
                        "end_index": cell_end,
                    }
                )
            rows_data.append({"cells": cells_data})

        return {
            "type": "table",
            "start_index": table_element.get("startIndex"),
            "end_index": table_element.get("endIndex"),
            "rows": rows_data,
            "row_count": len(table_rows),
            "column_count": table.get("columns", 0),
        }

    def _get_structure(self, service, document_id: str, detailed: bool) -> dict:
        doc = service.documents().get(documentId=document_id).execute()
        body = doc.get("body", {})
        content = body.get("content", [])

        segments: list[dict] = []
        structure_body: list[dict] = []

        for element in content:
            start_index = element.get("startIndex")
            end_index = element.get("endIndex")

            if "paragraph" in element:
                paragraph = element["paragraph"]
                text = self._extract_paragraph_text(paragraph)
                style_info = self._get_paragraph_style(paragraph)

                # Determine segment type
                if style_info.get("heading_level"):
                    seg_type = "heading"
                    segment = {
                        "type": seg_type,
                        "level": style_info["heading_level"],
                        "text": text,
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                else:
                    seg_type = "paragraph"
                    segment = {
                        "type": seg_type,
                        "text": text,
                        "start_index": start_index,
                        "end_index": end_index,
                    }

                # Skip empty paragraphs (just newlines)
                if text.strip():
                    segments.append(segment)

                if detailed:
                    detailed_seg = segment.copy()
                    detailed_seg["style"] = paragraph.get("paragraphStyle", {})
                    structure_body.append(detailed_seg)

            elif "table" in element:
                table = element.get("table", {})
                table_rows = table.get("tableRows", [])

                segment = {
                    "type": "table",
                    "rows": len(table_rows),
                    "columns": table.get("columns", 0),
                    "start_index": start_index,
                    "end_index": end_index,
                }
                segments.append(segment)

                if detailed:
                    structure_body.append(self._process_table_detailed(element))

            elif "sectionBreak" in element:
                # Skip section breaks in simple mode, include in detailed
                if detailed:
                    structure_body.append(
                        {
                            "type": "section_break",
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                    )

            elif "tableOfContents" in element:
                segment = {
                    "type": "table_of_contents",
                    "start_index": start_index,
                    "end_index": end_index,
                }
                segments.append(segment)

                if detailed:
                    structure_body.append(segment)

        result = {
            "segments": segments,
            "structure": {"body": structure_body} if detailed else {},
        }

        return result
