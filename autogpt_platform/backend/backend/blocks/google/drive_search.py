import asyncio
from datetime import datetime
from enum import Enum

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._drive_api import (
    DRIVE_READONLY_SCOPE,
    FILE_TYPE_CLAUSES,
    FOLDER_MIME_TYPE,
    LIST_FIELDS,
    DriveFile,
    GoogleFileType,
    build_drive_service,
    drive_error,
    parse_folder_id,
    quote_drive_value,
    to_drive_file,
)

_TEST_FILE_RESOURCE = {
    "id": "1a2b3c4d5e6f7g8h9i0j",
    "name": "Q3 Report",
    "mimeType": "application/vnd.google-apps.document",
    "webViewLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit",
    "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.document",
    "createdTime": "2026-07-01T09:00:00.000Z",
    "modifiedTime": "2026-09-20T15:30:00.000Z",
    "owners": [{"emailAddress": "owner@example.com"}],
    "parents": ["0AbCdEfGhIjKlUk9PVA"],
    "shared": True,
    "starred": False,
    "trashed": False,
}
_TEST_FILE = to_drive_file(_TEST_FILE_RESOURCE, TEST_CREDENTIALS.id)


class GoogleDriveSearchFilesBlock(Block):
    """Find files in Google Drive by name, content, type, folder or date."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [DRIVE_READONLY_SCOPE]
        )
        name_contains: str = SchemaField(
            description="Only files whose name contains this text",
            default="",
        )
        text_contains: str = SchemaField(
            description="Only files whose name or content contains this text",
            default="",
        )
        file_type: GoogleFileType = SchemaField(
            description="Only files of this type",
            default=GoogleFileType.ANY,
        )
        folder_id: str = SchemaField(
            description="Only files directly inside this folder (ID or URL; 'root' for My Drive)",
            default="",
        )
        modified_after: datetime | None = SchemaField(
            description="Only files modified after this time",
            default=None,
        )
        include_shared_drives: bool = SchemaField(
            description="Also search shared drives the user belongs to",
            default=True,
            advanced=True,
        )
        include_trashed: bool = SchemaField(
            description="Include files in the trash",
            default=False,
            advanced=True,
        )
        custom_query: str = SchemaField(
            description=(
                "Extra Drive search clause, ANDed with the filters above "
                "(Drive query syntax, e.g. \"'me' in owners\")"
            ),
            default="",
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of files to return",
            default=25,
            ge=1,
            le=1000,
        )
        page_token: str = SchemaField(
            description="Page token from a previous search, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        files: list[DriveFile] = SchemaField(
            description="Matching files, newest first unless searching content"
        )
        file: DriveFile = SchemaField(description="Each matching file")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="0575ebeb-6b51-40de-8772-1b83a64c7145",
            description=(
                "Search Google Drive for files by name, content, type, folder "
                "or modified date. Returns files you can pass to other Google blocks."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveSearchFilesBlock.Input,
            output_schema=GoogleDriveSearchFilesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "name_contains": "Q3 Report",
                "file_type": GoogleFileType.DOCUMENT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("files", [_TEST_FILE]),
                ("file", _TEST_FILE),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_list_files": lambda *args, **kwargs: {
                    "files": [_TEST_FILE_RESOURCE],
                    "nextPageToken": "next-page",
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        query = build_search_query(input_data)
        order_by = None if input_data.text_contains else "modifiedTime desc"
        service = build_drive_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._list_files,
                service,
                query=query,
                order_by=order_by,
                page_size=input_data.max_results,
                page_token=input_data.page_token,
                all_drives=input_data.include_shared_drives,
            )
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e

        files = [to_drive_file(f, credentials.id) for f in result.get("files", [])]
        yield "files", files
        for file in files:
            yield "file", file
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _list_files(
        service,
        *,
        query: str,
        order_by: str | None,
        page_size: int,
        page_token: str,
        all_drives: bool,
    ) -> dict:
        return list_drive_files(
            service,
            query=query,
            order_by=order_by,
            page_size=page_size,
            page_token=page_token,
            all_drives=all_drives,
        )


class RecentFilesOrder(str, Enum):
    RECENCY = "recency"
    LAST_MODIFIED = "last_modified"
    LAST_MODIFIED_BY_ME = "last_modified_by_me"
    LAST_VIEWED_BY_ME = "last_viewed_by_me"


_ORDER_BY: dict[RecentFilesOrder, str] = {
    RecentFilesOrder.RECENCY: "recency desc",
    RecentFilesOrder.LAST_MODIFIED: "modifiedTime desc",
    RecentFilesOrder.LAST_MODIFIED_BY_ME: "modifiedByMeTime desc",
    RecentFilesOrder.LAST_VIEWED_BY_ME: "viewedByMeTime desc",
}


class GoogleDriveListRecentFilesBlock(Block):
    """List the user's most recent Google Drive files."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [DRIVE_READONLY_SCOPE]
        )
        order_by: RecentFilesOrder = SchemaField(
            description="Which timestamp decides how recent a file is",
            default=RecentFilesOrder.RECENCY,
        )
        max_results: int = SchemaField(
            description="Maximum number of files to return",
            default=10,
            ge=1,
            le=1000,
        )
        page_token: str = SchemaField(
            description="Page token from a previous call, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        files: list[DriveFile] = SchemaField(description="Recent files, newest first")
        file: DriveFile = SchemaField(description="Each recent file")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="b5ac5ebc-7dcd-4203-b979-68ff5a72b4ac",
            description=(
                "List the most recently used, modified or viewed files in "
                "Google Drive. Folders are left out."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleDriveListRecentFilesBlock.Input,
            output_schema=GoogleDriveListRecentFilesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "max_results": 1},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("files", [_TEST_FILE]), ("file", _TEST_FILE)],
            test_mock={
                "_list_files": lambda *args, **kwargs: {"files": [_TEST_FILE_RESOURCE]}
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_drive_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._list_files,
                service,
                order_by=_ORDER_BY[input_data.order_by],
                page_size=input_data.max_results,
                page_token=input_data.page_token,
            )
        except HttpError as e:
            raise drive_error(e, self.name, self.id) from e

        files = [to_drive_file(f, credentials.id) for f in result.get("files", [])]
        yield "files", files
        for file in files:
            yield "file", file
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    @staticmethod
    def _list_files(service, *, order_by: str, page_size: int, page_token: str) -> dict:
        return list_drive_files(
            service,
            query=f"trashed = false and mimeType != '{FOLDER_MIME_TYPE}'",
            order_by=order_by,
            page_size=page_size,
            page_token=page_token,
            all_drives=False,
        )


def build_search_query(input_data: GoogleDriveSearchFilesBlock.Input) -> str:
    """Combine the search block's filters into one Drive query string."""
    clauses: list[str] = []
    if input_data.name_contains:
        clauses.append(f"name contains {quote_drive_value(input_data.name_contains)}")
    if input_data.text_contains:
        clauses.append(
            f"fullText contains {quote_drive_value(input_data.text_contains)}"
        )
    if input_data.file_type != GoogleFileType.ANY:
        clauses.append(FILE_TYPE_CLAUSES[input_data.file_type])
    if input_data.folder_id:
        folder = quote_drive_value(parse_folder_id(input_data.folder_id))
        clauses.append(f"{folder} in parents")
    if input_data.modified_after:
        clauses.append(f"modifiedTime > '{input_data.modified_after.isoformat()}'")
    if input_data.custom_query:
        clauses.append(f"({input_data.custom_query})")
    if not input_data.include_trashed:
        clauses.append("trashed = false")
    return " and ".join(clauses)


def list_drive_files(
    service,
    *,
    query: str,
    order_by: str | None,
    page_size: int,
    page_token: str,
    all_drives: bool,
) -> dict:
    params: dict = {
        "q": query,
        "pageSize": page_size,
        "fields": LIST_FIELDS,
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": all_drives,
        "corpora": "allDrives" if all_drives else "user",
    }
    if order_by:
        params["orderBy"] = order_by
    if page_token:
        params["pageToken"] = page_token
    return service.files().list(**params).execute()
