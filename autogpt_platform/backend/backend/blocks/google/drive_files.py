import asyncio

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.type import MediaFileType

from ._auth import GOOGLE_OAUTH_IS_CONFIGURED, TEST_CREDENTIALS, GoogleCredentials
from ._drive import GoogleDriveFile, GoogleDriveFileField
from ._drive_api import (
    DRIVE_READONLY_SCOPE,
    FILE_FIELDS,
    DriveFile,
    DrivePermission,
    build_drive_service,
    drive_error,
    require_file,
    to_drive_file,
)
from ._drive_content import (
    ExportFormat,
    fetch_file_bytes,
    file_name_for,
    read_as_text,
    save_to_file_store,
)

_TEST_PICKED_FILE = {
    "id": "1a2b3c4d5e6f7g8h9i0j",
    "name": "Q3 Report",
    "mimeType": "application/vnd.google-apps.document",
}
_TEST_METADATA = {
    **_TEST_PICKED_FILE,
    "webViewLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit",
    "modifiedTime": "2026-09-20T15:30:00.000Z",
    "owners": [{"emailAddress": "owner@example.com"}],
}
_TEST_FILE = to_drive_file(_TEST_METADATA, TEST_CREDENTIALS.id)


def _file_field(action: str) -> GoogleDriveFile:
    return GoogleDriveFileField(
        title="File",
        description=f"The Drive file to {action}",
        credentials_scopes=[DRIVE_READONLY_SCOPE],
    )


class GoogleDriveGetFileInfoBlock(Block):
    """Look up a Drive file's details: type, size, owners, dates, folder."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = _file_field("look up")

    class Output(BlockSchemaOutput):
        file: DriveFile = SchemaField(description="The file with its details")

    def __init__(self):
        super().__init__(
            id="b83913c6-6fe0-4a1a-bf0e-2da4b64eece5",
            description=(
                "Get a Google Drive file's details: name, type, size, owners, "
                "created and modified dates, and the folders it is in."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveGetFileInfoBlock.Input,
            output_schema=GoogleDriveGetFileInfoBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file", _TEST_FILE)],
            test_mock={"_get_metadata": lambda *args, **kwargs: _TEST_METADATA},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        service = build_drive_service(credentials)
        try:
            metadata = await asyncio.to_thread(self._get_metadata, service, file.id)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "file", to_drive_file(metadata, credentials.id)

    @staticmethod
    def _get_metadata(service, file_id: str) -> dict:
        return get_metadata(service, file_id)


class GoogleDriveReadFileBlock(Block):
    """Read a Drive file's content as text."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = _file_field("read")

    class Output(BlockSchemaOutput):
        content: str = SchemaField(
            description=(
                "The file as text: Markdown for Google Docs, CSV of the first "
                "sheet for Google Sheets, plain text for Slides, PDFs and text files"
            )
        )
        file: DriveFile = SchemaField(description="The file that was read")

    def __init__(self):
        super().__init__(
            id="c4c3ef5e-1169-4b05-95da-9c209408ae4f",
            description=(
                "Read a Google Drive file as text. Works for Google Docs, Sheets "
                "(first sheet), Slides, PDFs and plain-text files such as CSV, "
                "JSON or Markdown."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveReadFileBlock.Input,
            output_schema=GoogleDriveReadFileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("content", "# Q3 Report\n\nRevenue grew 12%."),
                ("file", _TEST_FILE),
            ],
            test_mock={
                "_get_metadata": lambda *args, **kwargs: _TEST_METADATA,
                "_read_text": lambda *args, **kwargs: "# Q3 Report\n\nRevenue grew 12%.",
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        service = build_drive_service(credentials)
        try:
            metadata = await asyncio.to_thread(self._get_metadata, service, file.id)
            content = await asyncio.to_thread(self._read_text, service, metadata)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "content", content
        yield "file", to_drive_file(metadata, credentials.id)

    @staticmethod
    def _get_metadata(service, file_id: str) -> dict:
        return get_metadata(service, file_id)

    @staticmethod
    def _read_text(service, metadata: dict) -> str:
        return read_as_text(service, metadata)


class GoogleDriveDownloadFileBlock(Block):
    """Download a Drive file, exporting Google Docs, Sheets and Slides."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = _file_field("download")
        export_format: ExportFormat = SchemaField(
            description=(
                "For Google Docs, Sheets and Slides: PDF, the matching Office "
                "format (DOCX, XLSX, PPTX), or text (Markdown, CSV, plain text). "
                "Other files download as they are."
            ),
            default=ExportFormat.PDF,
        )

    class Output(BlockSchemaOutput):
        content: MediaFileType = SchemaField(
            description="The downloaded file (a workspace file in CoPilot, a data URI in agents)"
        )
        mime_type: str = SchemaField(description="MIME type of the downloaded file")
        file: DriveFile = SchemaField(description="The Drive file that was downloaded")

    def __init__(self):
        super().__init__(
            id="74430172-7dc4-4e0a-b8f2-75c3d4090501",
            description=(
                "Download a file from Google Drive (up to 50 MB). Google Docs, "
                "Sheets and Slides are exported to PDF, Office or text formats."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveDownloadFileBlock.Input,
            output_schema=GoogleDriveDownloadFileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE, "export_format": ExportFormat.PDF},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("content", "data:application/pdf;base64,JVBERi0xLjQ="),
                ("mime_type", "application/pdf"),
                ("file", _TEST_FILE),
            ],
            test_mock={
                "_get_metadata": lambda *args, **kwargs: _TEST_METADATA,
                "_fetch_bytes": lambda *args, **kwargs: (
                    b"%PDF-1.4",
                    "application/pdf",
                ),
                "_store": lambda *args, **kwargs: "data:application/pdf;base64,JVBERi0xLjQ=",
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        service = build_drive_service(credentials)
        try:
            metadata = await asyncio.to_thread(self._get_metadata, service, file.id)
            data, mime_type = await asyncio.to_thread(
                self._fetch_bytes, service, metadata, input_data.export_format
            )
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        name = file_name_for(metadata.get("name") or file.id, mime_type)
        yield "content", await self._store(data, name, execution_context)
        yield "mime_type", mime_type
        yield "file", to_drive_file(metadata, credentials.id)

    @staticmethod
    def _get_metadata(service, file_id: str) -> dict:
        return get_metadata(service, file_id)

    @staticmethod
    def _fetch_bytes(
        service, metadata: dict, export_format: ExportFormat
    ) -> tuple[bytes, str]:
        return fetch_file_bytes(service, metadata, export_format)

    @staticmethod
    async def _store(
        data: bytes, name: str, execution_context: ExecutionContext
    ) -> MediaFileType:
        return await save_to_file_store(data, name, execution_context)


class GoogleDriveGetFilePermissionsBlock(Block):
    """List who can access a Drive file and with what role."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = _file_field("check")

    class Output(BlockSchemaOutput):
        permissions: list[DrivePermission] = SchemaField(
            description="Everyone who can access the file, with their role"
        )
        permission: DrivePermission = SchemaField(description="Each permission")

    def __init__(self):
        test_permission = DrivePermission(
            id="1234",
            type="user",
            role="writer",
            email_address="teammate@example.com",
            display_name="Teammate",
        )
        super().__init__(
            id="cfde953b-9d44-4c37-bd06-1b1c0144f527",
            description=(
                "List who can access a Google Drive file: users, groups, "
                "domains or anyone with the link, and their roles."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveGetFilePermissionsBlock.Input,
            output_schema=GoogleDriveGetFilePermissionsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("permissions", [test_permission]),
                ("permission", test_permission),
            ],
            test_mock={
                "_list_permissions": lambda *args, **kwargs: [
                    {
                        "id": "1234",
                        "type": "user",
                        "role": "writer",
                        "emailAddress": "teammate@example.com",
                        "displayName": "Teammate",
                    }
                ]
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        service = build_drive_service(credentials)
        try:
            items = await asyncio.to_thread(self._list_permissions, service, file.id)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        permissions = [
            DrivePermission(
                id=item["id"],
                type=item.get("type", ""),
                role=item.get("role", ""),
                email_address=item.get("emailAddress"),
                domain=item.get("domain"),
                display_name=item.get("displayName"),
            )
            for item in items
        ]
        yield "permissions", permissions
        for permission in permissions:
            yield "permission", permission

    @staticmethod
    def _list_permissions(service, file_id: str) -> list[dict]:
        permissions: list[dict] = []
        page_token = None
        while True:
            response = (
                service.permissions()
                .list(
                    fileId=file_id,
                    fields="nextPageToken, permissions(id, type, role, emailAddress, domain, displayName)",
                    supportsAllDrives=True,
                    pageToken=page_token,
                )
                .execute()
            )
            permissions.extend(response.get("permissions", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return permissions


def get_metadata(service, file_id: str) -> dict:
    return (
        service.files()
        .get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True)
        .execute()
    )
