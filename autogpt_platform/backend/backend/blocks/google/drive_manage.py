import asyncio
import mimetypes
from enum import Enum

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload
from pydantic import BaseModel

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError
from backend.util.type import MediaFileType

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._drive import GoogleDriveFile, GoogleDriveFileField
from ._drive_api import (
    DRIVE_FILE_SCOPE,
    DRIVE_SCOPE,
    FILE_FIELDS,
    FOLDER_MIME_TYPE,
    DriveFile,
    build_drive_service,
    drive_error,
    parse_folder_id,
    require_file,
    to_drive_file,
)
from ._drive_content import load_from_file_store


class ConvertTo(str, Enum):
    """Turn uploaded content into a Google file (e.g. CSV into a Google Sheet)."""

    NONE = "none"
    GOOGLE_DOC = "google_doc"
    GOOGLE_SHEET = "google_sheet"
    GOOGLE_SLIDES = "google_slides"


_GOOGLE_TYPES: dict[ConvertTo, str] = {
    ConvertTo.GOOGLE_DOC: "application/vnd.google-apps.document",
    ConvertTo.GOOGLE_SHEET: "application/vnd.google-apps.spreadsheet",
    ConvertTo.GOOGLE_SLIDES: "application/vnd.google-apps.presentation",
}


class _Upload(BaseModel):
    data: bytes | None = None
    content_type: str = ""
    name: str = ""


def _guess_type(name: str) -> str | None:
    return mimetypes.guess_type(name)[0] if name else None


_TEST_CREATED = {
    "id": "1n3wF1l3Id0123456789",
    "name": "notes.txt",
    "mimeType": "text/plain",
    "webViewLink": "https://drive.google.com/file/d/1n3wF1l3Id0123456789/view",
    "parents": ["0AbCdEfGhIjKlUk9PVA"],
}
_TEST_FOLDER = {
    "id": "1f0ld3rId0123456789",
    "name": "Invoices 2026",
    "mimeType": FOLDER_MIME_TYPE,
    "webViewLink": "https://drive.google.com/drive/folders/1f0ld3rId0123456789",
    "parents": ["0AbCdEfGhIjKlUk9PVA"],
}
_TEST_PICKED_FILE = {
    "id": "1a2b3c4d5e6f7g8h9i0j",
    "name": "Q3 Report",
    "mimeType": "application/vnd.google-apps.document",
}


class GoogleDriveCreateFileBlock(Block):
    """Create a file in Google Drive from text or an uploaded file."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField([DRIVE_FILE_SCOPE])
        name: str = SchemaField(
            description="Name of the new file, e.g. notes.txt. Defaults to the uploaded file's name.",
            default="",
        )
        text_content: str = SchemaField(
            description="Text to put in the file. Leave empty when uploading a file.",
            default="",
        )
        file_to_upload: MediaFileType | None = SchemaField(
            description="A file to upload instead of text (URL, data URI or workspace file)",
            default=None,
        )
        convert_to: ConvertTo = SchemaField(
            description=(
                "Convert the content into a Google Doc, Sheet or Slides file. "
                "With no content, creates an empty one."
            ),
            default=ConvertTo.NONE,
        )
        folder_id: str = SchemaField(
            description="Folder to create the file in (ID or URL). Empty means My Drive.",
            default="",
        )
        content_type: str = SchemaField(
            description="MIME type of the content. Worked out from the name when empty.",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        file: DriveFile = SchemaField(description="The new file")

    def __init__(self):
        super().__init__(
            id="fb2d6733-a429-4437-8e4a-3fe4fd40b233",
            description=(
                "Create a file in Google Drive from text or an uploaded file, "
                "optionally converting it to a Google Doc, Sheet or Slides file."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveCreateFileBlock.Input,
            output_schema=GoogleDriveCreateFileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "name": "notes.txt",
                "text_content": "Remember the milk",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file", to_drive_file(_TEST_CREATED, TEST_CREDENTIALS.id))],
            test_mock={"_create": lambda *args, **kwargs: _TEST_CREATED},
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        upload = await self._content(input_data, execution_context)
        name = input_data.name or upload.name
        if not name:
            raise BlockInputError(
                message="Give the new file a name.",
                block_name=self.name,
                block_id=self.id,
            )
        body: dict = {"name": name}
        if input_data.folder_id:
            body["parents"] = [parse_folder_id(input_data.folder_id)]
        if input_data.convert_to != ConvertTo.NONE:
            body["mimeType"] = _GOOGLE_TYPES[input_data.convert_to]
        elif upload.data is None:
            raise BlockInputError(
                message="Give the file some text content, a file to upload, or a Google type to convert to.",
                block_name=self.name,
                block_id=self.id,
            )
        service = build_drive_service(credentials)
        try:
            created = await asyncio.to_thread(
                self._create, service, body, upload.data, upload.content_type
            )
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "file", to_drive_file(created, credentials.id)

    async def _content(
        self, input_data: Input, execution_context: ExecutionContext
    ) -> _Upload:
        """Collect the bytes to upload (none for an empty Google file)."""
        if input_data.text_content and input_data.file_to_upload:
            raise BlockInputError(
                message="Set either text content or a file to upload, not both.",
                block_name=self.name,
                block_id=self.id,
            )
        if input_data.file_to_upload:
            upload_name, data = await load_from_file_store(
                input_data.file_to_upload, execution_context
            )
            guessed = _guess_type(input_data.name) or _guess_type(upload_name)
            return _Upload(
                data=data,
                content_type=input_data.content_type
                or guessed
                or "application/octet-stream",
                name=upload_name,
            )
        if input_data.text_content:
            return _Upload(
                data=input_data.text_content.encode("utf-8"),
                content_type=input_data.content_type
                or _guess_type(input_data.name)
                or "text/plain",
            )
        return _Upload()

    @staticmethod
    def _create(service, body: dict, data: bytes | None, content_type: str) -> dict:
        media = (
            MediaInMemoryUpload(data, mimetype=content_type, resumable=True)
            if data is not None
            else None
        )
        return (
            service.files()
            .create(
                body=body, media_body=media, fields=FILE_FIELDS, supportsAllDrives=True
            )
            .execute()
        )


class GoogleDriveCreateFolderBlock(Block):
    """Create a folder in Google Drive."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField([DRIVE_FILE_SCOPE])
        name: str = SchemaField(description="Name of the new folder")
        parent_folder_id: str = SchemaField(
            description="Folder to create it in (ID or URL). Empty means My Drive.",
            default="",
        )

    class Output(BlockSchemaOutput):
        folder: DriveFile = SchemaField(description="The new folder")

    def __init__(self):
        super().__init__(
            id="a4c31300-88c4-4336-acfd-3a651afb5ba6",
            description="Create a folder in Google Drive, optionally inside another folder.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveCreateFolderBlock.Input,
            output_schema=GoogleDriveCreateFolderBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "name": "Invoices 2026"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("folder", to_drive_file(_TEST_FOLDER, TEST_CREDENTIALS.id))],
            test_mock={"_create_folder": lambda *args, **kwargs: _TEST_FOLDER},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        body: dict = {"name": input_data.name, "mimeType": FOLDER_MIME_TYPE}
        if input_data.parent_folder_id:
            body["parents"] = [parse_folder_id(input_data.parent_folder_id)]
        service = build_drive_service(credentials)
        try:
            created = await asyncio.to_thread(self._create_folder, service, body)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "folder", to_drive_file(created, credentials.id)

    @staticmethod
    def _create_folder(service, body: dict) -> dict:
        return (
            service.files()
            .create(body=body, fields=FILE_FIELDS, supportsAllDrives=True)
            .execute()
        )


class GoogleDriveCopyFileBlock(Block):
    """Copy a Drive file, optionally renaming it or putting it in another folder."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = GoogleDriveFileField(
            title="File",
            description="The Drive file to copy",
            credentials_scopes=[DRIVE_SCOPE],
        )
        new_name: str = SchemaField(
            description="Name for the copy. Empty means 'Copy of <name>'.",
            default="",
        )
        folder_id: str = SchemaField(
            description="Folder for the copy (ID or URL). Empty means the original's folder.",
            default="",
        )

    class Output(BlockSchemaOutput):
        file: DriveFile = SchemaField(description="The copy")

    def __init__(self):
        copied = {
            **_TEST_CREATED,
            "id": "1c0pyId0123456789ab",
            "name": "Q3 Report (copy)",
        }
        super().__init__(
            id="75290430-1b29-4a2e-a3b7-bcd242aa1d56",
            description="Copy a Google Drive file, optionally with a new name or into another folder.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveCopyFileBlock.Input,
            output_schema=GoogleDriveCopyFileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"file": _TEST_PICKED_FILE, "new_name": "Q3 Report (copy)"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file", to_drive_file(copied, TEST_CREDENTIALS.id))],
            test_mock={"_copy": lambda *args, **kwargs: copied},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        body: dict = {}
        if input_data.new_name:
            body["name"] = input_data.new_name
        if input_data.folder_id:
            body["parents"] = [parse_folder_id(input_data.folder_id)]
        service = build_drive_service(credentials)
        try:
            copied = await asyncio.to_thread(self._copy, service, file.id, body)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "file", to_drive_file(copied, credentials.id)

    @staticmethod
    def _copy(service, file_id: str, body: dict) -> dict:
        return (
            service.files()
            .copy(fileId=file_id, body=body, fields=FILE_FIELDS, supportsAllDrives=True)
            .execute()
        )


class GoogleDriveMoveFileBlock(Block):
    """Move a Drive file into another folder."""

    class Input(BlockSchemaInput):
        file: GoogleDriveFile = GoogleDriveFileField(
            title="File",
            description="The Drive file to move",
            credentials_scopes=[DRIVE_SCOPE],
        )
        destination_folder_id: str = SchemaField(
            description="Folder to move it to (ID or URL; 'root' for My Drive)"
        )

    class Output(BlockSchemaOutput):
        file: DriveFile = SchemaField(description="The file in its new folder")

    def __init__(self):
        moved = {
            **_TEST_CREATED,
            **_TEST_PICKED_FILE,
            "parents": ["1f0ld3rId0123456789"],
        }
        super().__init__(
            id="8a1d80ca-780f-4c7a-bc19-01fc836eef79",
            description="Move a Google Drive file into another folder.",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveMoveFileBlock.Input,
            output_schema=GoogleDriveMoveFileBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "file": _TEST_PICKED_FILE,
                "destination_folder_id": "https://drive.google.com/drive/folders/1f0ld3rId0123456789",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("file", to_drive_file(moved, TEST_CREDENTIALS.id))],
            test_mock={"_move": lambda *args, **kwargs: moved},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        file = require_file(input_data.file, self.name, self.id)
        destination = parse_folder_id(input_data.destination_folder_id)
        service = build_drive_service(credentials)
        try:
            moved = await asyncio.to_thread(self._move, service, file.id, destination)
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e
        yield "file", to_drive_file(moved, credentials.id)

    @staticmethod
    def _move(service, file_id: str, destination: str) -> dict:
        current = (
            service.files()
            .get(fileId=file_id, fields="parents", supportsAllDrives=True)
            .execute()
        )
        return (
            service.files()
            .update(
                fileId=file_id,
                addParents=destination,
                removeParents=",".join(current.get("parents", [])),
                fields=FILE_FIELDS,
                supportsAllDrives=True,
            )
            .execute()
        )
