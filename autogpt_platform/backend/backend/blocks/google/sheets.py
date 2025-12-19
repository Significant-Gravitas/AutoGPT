import asyncio
import csv
import io
import re
from enum import Enum

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.blocks.google._drive import GoogleDriveFile, GoogleDriveFileField
from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
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
GOOGLE_SHEETS_DISABLED = not GOOGLE_OAUTH_IS_CONFIGURED


def parse_a1_notation(a1: str) -> tuple[str | None, str]:
    """Split an A1‑notation string into *(sheet_name, cell_range)*.

    Examples
    --------
    >>> parse_a1_notation("Sheet1!A1:B2")
    ("Sheet1", "A1:B2")
    >>> parse_a1_notation("A1:B2")
    (None, "A1:B2")
    """

    if "!" in a1:
        sheet, cell_range = a1.split("!", 1)
        return sheet, cell_range
    return None, a1


def extract_spreadsheet_id(spreadsheet_id_or_url: str) -> str:
    """Extract spreadsheet ID from either a direct ID or a Google Sheets URL.

    Examples
    --------
    >>> extract_spreadsheet_id("1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
    "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
    >>> extract_spreadsheet_id("https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit")
    "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
    """
    if "/spreadsheets/d/" in spreadsheet_id_or_url:
        # Extract ID from URL: https://docs.google.com/spreadsheets/d/{ID}/edit...
        parts = spreadsheet_id_or_url.split("/d/")[1].split("/")[0]
        return parts
    return spreadsheet_id_or_url


def format_sheet_name(sheet_name: str) -> str:
    """Format sheet name for Google Sheets API, adding quotes if needed.

    Examples
    --------
    >>> format_sheet_name("Sheet1")
    "Sheet1"
    >>> format_sheet_name("Non-matching Leads")
    "'Non-matching Leads'"
    """
    # If sheet name contains spaces, special characters, or starts with a digit, wrap in quotes
    if (
        " " in sheet_name
        or any(char in sheet_name for char in "!@#$%^&*()+-=[]{}|;:,.<>?")
        or (sheet_name and sheet_name[0].isdigit())
    ):
        return f"'{sheet_name}'"
    return sheet_name


def _first_sheet_meta(service, spreadsheet_id: str) -> tuple[str, int]:
    """Return *(title, sheetId)* for the first sheet in *spreadsheet_id*."""

    meta = (
        service.spreadsheets()
        .get(spreadsheetId=spreadsheet_id, includeGridData=False)
        .execute()
    )
    first = meta["sheets"][0]["properties"]
    return first["title"], first["sheetId"]


def get_all_sheet_names(service, spreadsheet_id: str) -> list[str]:
    """Get all sheet names in the spreadsheet."""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [
        sheet.get("properties", {}).get("title", "") for sheet in meta.get("sheets", [])
    ]


def resolve_sheet_name(service, spreadsheet_id: str, sheet_name: str | None) -> str:
    """Resolve *sheet_name*, falling back to the workbook's first sheet if empty.

    Validates that the sheet exists in the spreadsheet and provides helpful error info.
    """
    if sheet_name:
        # Validate that the sheet exists
        all_sheets = get_all_sheet_names(service, spreadsheet_id)
        if sheet_name not in all_sheets:
            raise ValueError(
                f'Sheet "{sheet_name}" not found in spreadsheet. '
                f"Available sheets: {all_sheets}"
            )
        return sheet_name
    title, _ = _first_sheet_meta(service, spreadsheet_id)
    return title


def sheet_id_by_name(service, spreadsheet_id: str, sheet_name: str) -> int | None:
    """Return the *sheetId* for *sheet_name* (or `None` if not found)."""

    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sh in meta.get("sheets", []):
        if sh.get("properties", {}).get("title") == sheet_name:
            return sh["properties"]["sheetId"]
    return None


def _build_sheets_service(credentials: GoogleCredentials):
    """Build Sheets service from platform credentials (with refresh token)."""
    settings = Settings()
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
    return build("sheets", "v4", credentials=creds)


def _build_drive_service(credentials: GoogleCredentials):
    """Build Drive service from platform credentials (with refresh token)."""
    settings = Settings()
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
    return build("drive", "v3", credentials=creds)


def _validate_spreadsheet_file(spreadsheet_file: "GoogleDriveFile") -> str | None:
    """Validate that the selected file is a Google Sheets spreadsheet.

    Returns None if valid, error message string if invalid.
    """
    if spreadsheet_file.mime_type != "application/vnd.google-apps.spreadsheet":
        file_type = spreadsheet_file.mime_type
        file_name = spreadsheet_file.name
        if file_type == "text/csv":
            return f"Cannot use CSV file '{file_name}' with Google Sheets block. Please use a CSV reader block instead, or convert the CSV to a Google Sheets spreadsheet first."
        elif file_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
            return f"Cannot use Excel file '{file_name}' with Google Sheets block. Please use an Excel reader block instead, or convert to Google Sheets first."
        else:
            return f"Cannot use file '{file_name}' (type: {file_type}) with Google Sheets block. This block only works with Google Sheets spreadsheets."
    return None


def _handle_sheets_api_error(error_msg: str, operation: str = "access") -> str:
    """Convert common Google Sheets API errors to user-friendly messages."""
    if "Request contains an invalid argument" in error_msg:
        return f"Invalid request to Google Sheets API. This usually means the file is not a Google Sheets spreadsheet, the range is invalid, or you don't have permission to {operation} this file."
    elif "The caller does not have permission" in error_msg or "Forbidden" in error_msg:
        if operation in ["write", "modify", "update", "append", "clear"]:
            return "Permission denied. You don't have edit access to this spreadsheet. Make sure it's shared with edit permissions."
        else:
            return "Permission denied. You don't have access to this spreadsheet. Make sure it's shared with you and try re-selecting the file."
    elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
        return "Spreadsheet not found. The file may have been deleted or the link is invalid."
    else:
        return f"Failed to {operation} Google Sheet: {error_msg}"


class SheetOperation(str, Enum):
    CREATE = "create"
    DELETE = "delete"
    COPY = "copy"


class ValueInputOption(str, Enum):
    RAW = "RAW"
    USER_ENTERED = "USER_ENTERED"


class InsertDataOption(str, Enum):
    OVERWRITE = "OVERWRITE"
    INSERT_ROWS = "INSERT_ROWS"


class BatchOperationType(str, Enum):
    UPDATE = "update"
    CLEAR = "clear"


class PublicAccessRole(str, Enum):
    READER = "reader"
    COMMENTER = "commenter"


class ShareRole(str, Enum):
    READER = "reader"
    WRITER = "writer"
    COMMENTER = "commenter"


class BatchOperation(BlockSchemaInput):
    type: BatchOperationType = SchemaField(
        description="The type of operation to perform"
    )
    range: str = SchemaField(description="The A1 notation range for the operation")
    values: list[list[str]] = SchemaField(
        description="Values to update (only for UPDATE)", default=[]
    )


class GoogleSheetsReadBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        range: str = SchemaField(
            description="The A1 notation of the range to read",
            placeholder="Sheet1!A1:Z1000",
        )

    class Output(BlockSchemaOutput):
        result: list[list[str]] = SchemaField(
            description="The data read from the spreadsheet",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="5724e902-3635-47e9-a108-aaa0263a4988",
            description="This block reads data from a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsReadBlock.Input,
            output_schema=GoogleSheetsReadBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "Sheet1!A1:B2",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    [
                        ["Name", "Score"],
                        ["Alice", "85"],
                    ],
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_read_sheet": lambda *args, **kwargs: [
                    ["Name", "Score"],
                    ["Alice", "85"],
                ],
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return
        try:
            service = _build_sheets_service(credentials)
            spreadsheet_id = input_data.spreadsheet.id
            data = await asyncio.to_thread(
                self._read_sheet, service, spreadsheet_id, input_data.range
            )
            yield "result", data
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=spreadsheet_id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", _handle_sheets_api_error(str(e), "read")

    def _read_sheet(self, service, spreadsheet_id: str, range: str) -> list[list[str]]:
        sheet = service.spreadsheets()
        range_to_use = range or "A:Z"
        sheet_name, cell_range = parse_a1_notation(range_to_use)
        if sheet_name:
            cleaned_sheet = sheet_name.strip().strip("'\"")
            formatted_sheet = format_sheet_name(cleaned_sheet)
            cell_part = cell_range.strip() if cell_range else ""
            if cell_part:
                range_to_use = f"{formatted_sheet}!{cell_part}"
            else:
                range_to_use = f"{formatted_sheet}!A:Z"
        # If no sheet name, keep the original range (e.g., "A1:B2" or "B:B")
        result = (
            sheet.values()
            .get(spreadsheetId=spreadsheet_id, range=range_to_use)
            .execute()
        )
        return result.get("values", [])


class GoogleSheetsWriteBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        range: str = SchemaField(
            description="The A1 notation of the range to write",
            placeholder="Sheet1!A1:B2",
        )
        values: list[list[str]] = SchemaField(
            description="The data to write to the spreadsheet",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the write operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="d9291e87-301d-47a8-91fe-907fb55460e5",
            description="This block writes data to a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsWriteBlock.Input,
            output_schema=GoogleSheetsWriteBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "Sheet1!A1:B2",
                "values": [
                    ["Name", "Score"],
                    ["Bob", "90"],
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"updatedCells": 4, "updatedColumns": 2, "updatedRows": 2},
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_write_sheet": lambda *args, **kwargs: {
                    "updatedCells": 4,
                    "updatedColumns": 2,
                    "updatedRows": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            # Customize message for write operations on CSV files
            if "CSV file" in validation_error:
                yield "error", validation_error.replace(
                    "Please use a CSV reader block instead, or",
                    "CSV files are read-only through Google Drive. Please",
                )
            else:
                yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._write_sheet,
                service,
                input_data.spreadsheet.id,
                input_data.range,
                input_data.values,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", _handle_sheets_api_error(str(e), "write")

    def _write_sheet(
        self, service, spreadsheet_id: str, range: str, values: list[list[str]]
    ) -> dict:
        body = {"values": values}
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=range,
                valueInputOption="USER_ENTERED",
                body=body,
            )
            .execute()
        )
        return result


class GoogleSheetsAppendRowBlock(Block):
    """Append a single row to the end of a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        row: list[str] = SchemaField(
            description="Row values to append (e.g., ['Alice', 'alice@example.com', '25'])",
        )
        sheet_name: str = SchemaField(
            description="Sheet to append to (optional, defaults to first sheet)",
            default="",
        )
        value_input_option: ValueInputOption = SchemaField(
            description="How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing.",
            default=ValueInputOption.USER_ENTERED,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Append API response")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining to other blocks",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="531d50c0-d6b9-4cf9-a013-7bf783d313c7",
            description="Append or Add a single row to the end of a Google Sheet. The row is added after the last row with data.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAppendRowBlock.Input,
            output_schema=GoogleSheetsAppendRowBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "row": ["Charlie", "95"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"updatedCells": 2, "updatedColumns": 2, "updatedRows": 1}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_append_row": lambda *args, **kwargs: {
                    "updatedCells": 2,
                    "updatedColumns": 2,
                    "updatedRows": 1,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.row:
            yield "error", "Row data is required"
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._append_row,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.row,
                input_data.value_input_option,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to append row: {str(e)}"

    def _append_row(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        row: list[str],
        value_input_option: ValueInputOption,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)
        append_range = f"{formatted_sheet}!A1"
        body = {"values": [row]}  # Wrap single row in list for API
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=append_range,
                valueInputOption=value_input_option.value,
                insertDataOption="INSERT_ROWS",
                body=body,
            )
            .execute()
        )
        return {
            "updatedCells": result.get("updates", {}).get("updatedCells", 0),
            "updatedRows": result.get("updates", {}).get("updatedRows", 0),
            "updatedColumns": result.get("updates", {}).get("updatedColumns", 0),
        }


class GoogleSheetsClearBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        range: str = SchemaField(
            description="The A1 notation of the range to clear",
            placeholder="Sheet1!A1:B2",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the clear operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="84938266-0fc7-46e5-9369-adb0f6ae8015",
            description="This block clears data from a specified range in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsClearBlock.Input,
            output_schema=GoogleSheetsClearBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "Sheet1!A1:B2",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"clearedRange": "Sheet1!A1:B2"}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_clear_range": lambda *args, **kwargs: {
                    "clearedRange": "Sheet1!A1:B2"
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._clear_range,
                service,
                input_data.spreadsheet.id,
                input_data.range,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to clear Google Sheet range: {str(e)}"

    def _clear_range(self, service, spreadsheet_id: str, range: str) -> dict:
        result = (
            service.spreadsheets()
            .values()
            .clear(spreadsheetId=spreadsheet_id, range=range)
            .execute()
        )
        return result


class GoogleSheetsMetadataBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The metadata of the spreadsheet including sheets info",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="6a0be6ee-7a0d-4c92-819b-500630846ad0",
            description="This block retrieves metadata about a Google Sheets spreadsheet including sheet names and properties.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsMetadataBlock.Input,
            output_schema=GoogleSheetsMetadataBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {
                        "title": "Test Spreadsheet",
                        "sheets": [{"title": "Sheet1", "sheetId": 0}],
                    },
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_metadata": lambda *args, **kwargs: {
                    "title": "Test Spreadsheet",
                    "sheets": [{"title": "Sheet1", "sheetId": 0}],
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_metadata,
                service,
                input_data.spreadsheet.id,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get spreadsheet metadata: {str(e)}"

    def _get_metadata(self, service, spreadsheet_id: str) -> dict:
        result = (
            service.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, includeGridData=False)
            .execute()
        )
        return {
            "title": result.get("properties", {}).get("title"),
            "sheets": [
                {
                    "title": sheet.get("properties", {}).get("title"),
                    "sheetId": sheet.get("properties", {}).get("sheetId"),
                    "gridProperties": sheet.get("properties", {}).get(
                        "gridProperties", {}
                    ),
                }
                for sheet in result.get("sheets", [])
            ],
        }


class GoogleSheetsManageSheetBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        operation: SheetOperation = SchemaField(description="Operation to perform")
        sheet_name: str = SchemaField(
            description="Target sheet name (defaults to first sheet for delete)",
            default="",
        )
        source_sheet_id: int = SchemaField(
            description="Source sheet ID for copy", default=0
        )
        destination_sheet_name: str = SchemaField(
            description="New sheet name for copy", default=""
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Operation result")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="7940189d-b137-4ef1-aa18-3dd9a5bde9f3",
            description="Create, delete, or copy sheets (sheet optional)",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsManageSheetBlock.Input,
            output_schema=GoogleSheetsManageSheetBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "operation": SheetOperation.CREATE,
                "sheet_name": "NewSheet",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "sheetId": 123}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_manage_sheet": lambda *args, **kwargs: {
                    "success": True,
                    "sheetId": 123,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._manage_sheet,
                service,
                input_data.spreadsheet.id,
                input_data.operation,
                input_data.sheet_name,
                input_data.source_sheet_id,
                input_data.destination_sheet_name,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to manage sheet: {str(e)}"

    def _manage_sheet(
        self,
        service,
        spreadsheet_id: str,
        operation: SheetOperation,
        sheet_name: str,
        source_sheet_id: int,
        destination_sheet_name: str,
    ) -> dict:
        requests = []

        if operation == SheetOperation.CREATE:
            # For CREATE, use sheet_name directly or default to "New Sheet"
            target_name = sheet_name or "New Sheet"
            requests.append({"addSheet": {"properties": {"title": target_name}}})
        elif operation == SheetOperation.DELETE:
            # For DELETE, resolve sheet name (fall back to first sheet if empty)
            target_name = resolve_sheet_name(
                service, spreadsheet_id, sheet_name or None
            )
            sid = sheet_id_by_name(service, spreadsheet_id, target_name)
            if sid is None:
                return {"error": f"Sheet '{target_name}' not found"}
            requests.append({"deleteSheet": {"sheetId": sid}})
        elif operation == SheetOperation.COPY:
            # For COPY, use source_sheet_id and destination_sheet_name directly
            requests.append(
                {
                    "duplicateSheet": {
                        "sourceSheetId": source_sheet_id,
                        "newSheetName": destination_sheet_name
                        or f"Copy of {source_sheet_id}",
                    }
                }
            )
        else:
            return {"error": f"Unknown operation: {operation}"}

        body = {"requests": requests}
        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )
        return {"success": True, "result": result}


class GoogleSheetsBatchOperationsBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        operations: list[BatchOperation] = SchemaField(
            description="List of operations to perform",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the batch operations",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="a4078584-6fe5-46e0-997e-d5126cdd112a",
            description="This block performs multiple operations on a Google Sheets spreadsheet in a single batch request.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsBatchOperationsBlock.Input,
            output_schema=GoogleSheetsBatchOperationsBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "operations": [
                    {
                        "type": BatchOperationType.UPDATE,
                        "range": "A1:B1",
                        "values": [["Header1", "Header2"]],
                    },
                    {
                        "type": BatchOperationType.UPDATE,
                        "range": "A2:B2",
                        "values": [["Data1", "Data2"]],
                    },
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"totalUpdatedCells": 4, "replies": []}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_batch_operations": lambda *args, **kwargs: {
                    "totalUpdatedCells": 4,
                    "replies": [],
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._batch_operations,
                service,
                input_data.spreadsheet.id,
                input_data.operations,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to perform batch operations: {str(e)}"

    def _batch_operations(
        self, service, spreadsheet_id: str, operations: list[BatchOperation]
    ) -> dict:
        update_data = []
        clear_ranges = []

        for op in operations:
            if op.type == BatchOperationType.UPDATE:
                update_data.append(
                    {
                        "range": op.range,
                        "values": op.values,
                    }
                )
            elif op.type == BatchOperationType.CLEAR:
                clear_ranges.append(op.range)

        results = {}

        # Perform updates if any
        if update_data:
            update_body = {
                "valueInputOption": "USER_ENTERED",
                "data": update_data,
            }
            update_result = (
                service.spreadsheets()
                .values()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=update_body)
                .execute()
            )
            results["updateResult"] = update_result

        # Perform clears if any
        if clear_ranges:
            clear_body = {"ranges": clear_ranges}
            clear_result = (
                service.spreadsheets()
                .values()
                .batchClear(spreadsheetId=spreadsheet_id, body=clear_body)
                .execute()
            )
            results["clearResult"] = clear_result

        return results


class GoogleSheetsFindReplaceBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        find_text: str = SchemaField(
            description="The text to find",
        )
        replace_text: str = SchemaField(
            description="The text to replace with",
        )
        sheet_id: int = SchemaField(
            description="The ID of the specific sheet to search (optional, searches all sheets if not provided)",
            default=-1,
        )
        match_case: bool = SchemaField(
            description="Whether to match case",
            default=False,
        )
        match_entire_cell: bool = SchemaField(
            description="Whether to match entire cell",
            default=False,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the find/replace operation including number of replacements",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="accca760-8174-4656-b55e-5f0e82fee986",
            description="This block finds and replaces text in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFindReplaceBlock.Input,
            output_schema=GoogleSheetsFindReplaceBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "find_text": "old_value",
                "replace_text": "new_value",
                "match_case": False,
                "match_entire_cell": False,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"occurrencesChanged": 5}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_find_replace": lambda *args, **kwargs: {"occurrencesChanged": 5},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._find_replace,
                service,
                input_data.spreadsheet.id,
                input_data.find_text,
                input_data.replace_text,
                input_data.sheet_id,
                input_data.match_case,
                input_data.match_entire_cell,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to find/replace in Google Sheet: {str(e)}"

    def _find_replace(
        self,
        service,
        spreadsheet_id: str,
        find_text: str,
        replace_text: str,
        sheet_id: int,
        match_case: bool,
        match_entire_cell: bool,
    ) -> dict:
        find_replace_request = {
            "find": find_text,
            "replacement": replace_text,
            "matchCase": match_case,
            "matchEntireCell": match_entire_cell,
        }

        if sheet_id >= 0:
            find_replace_request["sheetId"] = sheet_id

        requests = [{"findReplace": find_replace_request}]
        body = {"requests": requests}

        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
            .execute()
        )

        return result


class GoogleSheetsFindBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        find_text: str = SchemaField(
            description="The text to find",
        )
        sheet_id: int = SchemaField(
            description="The ID of the specific sheet to search (optional, searches all sheets if not provided)",
            default=-1,
        )
        match_case: bool = SchemaField(
            description="Whether to match case",
            default=False,
        )
        match_entire_cell: bool = SchemaField(
            description="Whether to match entire cell",
            default=False,
        )
        find_all: bool = SchemaField(
            description="Whether to find all occurrences (true) or just the first one (false)",
            default=True,
        )
        range: str = SchemaField(
            description="The A1 notation range to search in (optional, searches entire sheet if not provided)",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the find operation including locations and count",
        )
        locations: list[dict] = SchemaField(
            description="List of cell locations where the text was found",
        )
        count: int = SchemaField(
            description="Number of occurrences found",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="0f4ecc72-b958-47b2-b65e-76d6d26b9b27",
            description="Find text in a Google Sheets spreadsheet. Returns locations and count of occurrences. Can find all occurrences or just the first one.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFindBlock.Input,
            output_schema=GoogleSheetsFindBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "find_text": "search_value",
                "match_case": False,
                "match_entire_cell": False,
                "find_all": True,
                "range": "Sheet1!A1:C10",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("count", 3),
                (
                    "locations",
                    [
                        {"sheet": "Sheet1", "row": 2, "column": 1, "address": "A2"},
                        {"sheet": "Sheet1", "row": 5, "column": 3, "address": "C5"},
                        {"sheet": "Sheet2", "row": 1, "column": 2, "address": "B1"},
                    ],
                ),
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_find_text": lambda *args, **kwargs: {
                    "locations": [
                        {"sheet": "Sheet1", "row": 2, "column": 1, "address": "A2"},
                        {"sheet": "Sheet1", "row": 5, "column": 3, "address": "C5"},
                        {"sheet": "Sheet2", "row": 1, "column": 2, "address": "B1"},
                    ],
                    "count": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._find_text,
                service,
                input_data.spreadsheet.id,
                input_data.find_text,
                input_data.sheet_id,
                input_data.match_case,
                input_data.match_entire_cell,
                input_data.find_all,
                input_data.range,
            )
            yield "count", result["count"]
            yield "locations", result["locations"]
            yield "result", {"success": True}
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to find text in Google Sheet: {str(e)}"

    def _find_text(
        self,
        service,
        spreadsheet_id: str,
        find_text: str,
        sheet_id: int,
        match_case: bool,
        match_entire_cell: bool,
        find_all: bool,
        range: str,
    ) -> dict:
        # Unfortunately, Google Sheets API doesn't have a dedicated "find-only" operation
        # that returns cell locations. The findReplace operation only returns a count.
        # So we need to search through the values manually to get location details.

        locations = []
        search_range = range if range else None

        if not search_range:
            # If no range specified, search entire spreadsheet
            meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheets = meta.get("sheets", [])

            # Filter to specific sheet if provided
            if sheet_id >= 0:
                sheets = [
                    s
                    for s in sheets
                    if s.get("properties", {}).get("sheetId") == sheet_id
                ]

            # Search each sheet
            for sheet in sheets:
                sheet_name = sheet.get("properties", {}).get("title", "")
                sheet_range = f"'{sheet_name}'"
                self._search_range(
                    service,
                    spreadsheet_id,
                    sheet_range,
                    sheet_name,
                    find_text,
                    match_case,
                    match_entire_cell,
                    find_all,
                    locations,
                )
                if not find_all and locations:
                    break
        else:
            # Search specific range
            sheet_name, cell_range = parse_a1_notation(search_range)
            if not sheet_name:
                # Get first sheet name if not specified
                meta = (
                    service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
                )
                sheet_name = (
                    meta.get("sheets", [{}])[0]
                    .get("properties", {})
                    .get("title", "Sheet1")
                )
                search_range = f"'{sheet_name}'!{search_range}"

            self._search_range(
                service,
                spreadsheet_id,
                search_range,
                sheet_name,
                find_text,
                match_case,
                match_entire_cell,
                find_all,
                locations,
            )

        return {"locations": locations, "count": len(locations)}

    def _search_range(
        self,
        service,
        spreadsheet_id: str,
        range_name: str,
        sheet_name: str,
        find_text: str,
        match_case: bool,
        match_entire_cell: bool,
        find_all: bool,
        locations: list,
    ):
        """Search within a specific range and add results to locations list."""
        values_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )
        values = values_result.get("values", [])

        # Parse range to get starting position
        _, cell_range = parse_a1_notation(range_name)
        start_col = 0
        start_row = 0

        if cell_range and ":" in cell_range:
            start_cell = cell_range.split(":")[0]
            # Parse A1 notation (e.g., "B3" -> col=1, row=2)
            col_part = ""
            row_part = ""
            for char in start_cell:
                if char.isalpha():
                    col_part += char
                elif char.isdigit():
                    row_part += char

            if col_part:
                start_col = ord(col_part.upper()) - ord("A")
            if row_part:
                start_row = int(row_part) - 1

        # Search through values
        for row_idx, row in enumerate(values):
            for col_idx, cell_value in enumerate(row):
                if cell_value is None:
                    continue

                cell_str = str(cell_value)

                # Apply search criteria
                search_text = find_text if match_case else find_text.lower()
                cell_text = cell_str if match_case else cell_str.lower()

                found = False
                if match_entire_cell:
                    found = cell_text == search_text
                else:
                    found = search_text in cell_text

                if found:
                    # Calculate actual spreadsheet position
                    actual_row = start_row + row_idx + 1
                    actual_col = start_col + col_idx + 1
                    col_letter = chr(ord("A") + start_col + col_idx)
                    address = f"{col_letter}{actual_row}"

                    location = {
                        "sheet": sheet_name,
                        "row": actual_row,
                        "column": actual_col,
                        "address": address,
                        "value": cell_str,
                    }
                    locations.append(location)

                    # Stop after first match if find_all is False
                    if not find_all:
                        return


class GoogleSheetsFormatBlock(Block):
    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        range: str = SchemaField(
            description="A1 notation – sheet optional",
            placeholder="Sheet1!A1:B2",
        )
        background_color: dict = SchemaField(default={})
        text_color: dict = SchemaField(default={})
        bold: bool = SchemaField(default=False)
        italic: bool = SchemaField(default=False)
        font_size: int = SchemaField(default=10)

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="API response or success flag")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="270f2384-8089-4b5b-b2e3-fe2ea3d87c02",
            description="Format a range in a Google Sheet (sheet optional)",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFormatBlock.Input,
            output_schema=GoogleSheetsFormatBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "A1:B2",
                "background_color": {"red": 1.0, "green": 0.9, "blue": 0.9},
                "bold": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={"_format_cells": lambda *args, **kwargs: {"success": True}},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._format_cells,
                service,
                input_data.spreadsheet.id,
                input_data.range,
                input_data.background_color,
                input_data.text_color,
                input_data.bold,
                input_data.italic,
                input_data.font_size,
            )
            if "error" in result:
                yield "error", result["error"]
            else:
                yield "result", result
                # Output the GoogleDriveFile for chaining (preserves credentials_id)
                yield "spreadsheet", GoogleDriveFile(
                    id=input_data.spreadsheet.id,
                    name=input_data.spreadsheet.name,
                    mimeType="application/vnd.google-apps.spreadsheet",
                    url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                    iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                    isFolder=False,
                    _credentials_id=input_data.spreadsheet.credentials_id,
                )
        except Exception as e:
            yield "error", f"Failed to format Google Sheet cells: {str(e)}"

    def _format_cells(
        self,
        service,
        spreadsheet_id: str,
        a1_range: str,
        background_color: dict,
        text_color: dict,
        bold: bool,
        italic: bool,
        font_size: int,
    ) -> dict:
        sheet_name, cell_range = parse_a1_notation(a1_range)
        sheet_name = resolve_sheet_name(service, spreadsheet_id, sheet_name)

        sheet_id = sheet_id_by_name(service, spreadsheet_id, sheet_name)
        if sheet_id is None:
            return {"error": f"Sheet '{sheet_name}' not found"}

        try:
            start_cell, end_cell = cell_range.split(":")
            start_col = ord(start_cell[0].upper()) - ord("A")
            start_row = int(start_cell[1:]) - 1
            end_col = ord(end_cell[0].upper()) - ord("A") + 1
            end_row = int(end_cell[1:])
        except (ValueError, IndexError):
            return {"error": f"Invalid range format: {a1_range}"}

        cell_format: dict = {"userEnteredFormat": {}}
        if background_color:
            cell_format["userEnteredFormat"]["backgroundColor"] = background_color

        text_format: dict = {}
        if text_color:
            text_format["foregroundColor"] = text_color
        if bold:
            text_format["bold"] = True
        if italic:
            text_format["italic"] = True
        if font_size != 10:
            text_format["fontSize"] = font_size
        if text_format:
            cell_format["userEnteredFormat"]["textFormat"] = text_format

        body = {
            "requests": [
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": start_row,
                            "endRowIndex": end_row,
                            "startColumnIndex": start_col,
                            "endColumnIndex": end_col,
                        },
                        "cell": cell_format,
                        "fields": "userEnteredFormat(backgroundColor,textFormat)",
                    }
                }
            ]
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body=body
        ).execute()
        return {"success": True}


class GoogleSheetsCreateSpreadsheetBlock(Block):
    class Input(BlockSchemaInput):
        # Explicit credentials since this block creates a file (no file picker)
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/drive.file"]
        )
        title: str = SchemaField(
            description="The title of the new spreadsheet",
        )
        sheet_names: list[str] = SchemaField(
            description="List of sheet names to create (optional, defaults to single 'Sheet1')",
            default=["Sheet1"],
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result containing spreadsheet ID and URL",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The created spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the created spreadsheet",
        )
        spreadsheet_url: str = SchemaField(
            description="The URL of the created spreadsheet",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="c8d4c0d3-c76e-4c2a-8c66-4119817ea3d1",
            description="This block creates a new Google Sheets spreadsheet with specified sheets.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsCreateSpreadsheetBlock.Input,
            output_schema=GoogleSheetsCreateSpreadsheetBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "title": "Test Spreadsheet",
                "sheet_names": ["Sheet1", "Data", "Summary"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=TEST_CREDENTIALS_INPUT[
                            "id"
                        ],  # Preserves credential ID for chaining
                    ),
                ),
                ("spreadsheet_id", "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"),
                (
                    "spreadsheet_url",
                    "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                ),
                ("result", {"success": True}),
            ],
            test_mock={
                "_create_spreadsheet": lambda *args, **kwargs: {
                    "spreadsheetId": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                    "title": "Test Spreadsheet",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        drive_service = _build_drive_service(credentials)
        sheets_service = _build_sheets_service(credentials)
        result = await asyncio.to_thread(
            self._create_spreadsheet,
            drive_service,
            sheets_service,
            input_data.title,
            input_data.sheet_names,
        )

        if "error" in result:
            yield "error", result["error"]
        else:
            spreadsheet_id = result["spreadsheetId"]
            spreadsheet_url = result["spreadsheetUrl"]
            # Output the GoogleDriveFile for chaining (includes credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=spreadsheet_id,
                name=result.get("title", input_data.title),
                mimeType="application/vnd.google-apps.spreadsheet",
                url=spreadsheet_url,
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.credentials.id,  # Preserve credentials for chaining
            )
            yield "spreadsheet_id", spreadsheet_id
            yield "spreadsheet_url", spreadsheet_url
            yield "result", {"success": True}

    def _create_spreadsheet(
        self, drive_service, sheets_service, title: str, sheet_names: list[str]
    ) -> dict:
        try:
            # Create blank spreadsheet using Drive API
            file_metadata = {
                "name": title,
                "mimeType": "application/vnd.google-apps.spreadsheet",
            }
            result = (
                drive_service.files()
                .create(body=file_metadata, fields="id, webViewLink")
                .execute()
            )

            spreadsheet_id = result["id"]
            spreadsheet_url = result.get(
                "webViewLink",
                f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            )

            # Rename first sheet if custom name provided (default is "Sheet1")
            if sheet_names and sheet_names[0] != "Sheet1":
                # Get first sheet ID and rename it
                meta = (
                    sheets_service.spreadsheets()
                    .get(spreadsheetId=spreadsheet_id)
                    .execute()
                )
                first_sheet_id = meta["sheets"][0]["properties"]["sheetId"]
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={
                        "requests": [
                            {
                                "updateSheetProperties": {
                                    "properties": {
                                        "sheetId": first_sheet_id,
                                        "title": sheet_names[0],
                                    },
                                    "fields": "title",
                                }
                            }
                        ]
                    },
                ).execute()

            # Add additional sheets if requested
            if len(sheet_names) > 1:
                requests = [
                    {"addSheet": {"properties": {"title": name}}}
                    for name in sheet_names[1:]
                ]
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id, body={"requests": requests}
                ).execute()

            return {
                "spreadsheetId": spreadsheet_id,
                "spreadsheetUrl": spreadsheet_url,
                "title": title,
            }
        except Exception as e:
            return {"error": str(e)}


class GoogleSheetsUpdateCellBlock(Block):
    """Update a single cell in a Google Sheets spreadsheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        cell: str = SchemaField(
            description="Cell address in A1 notation (e.g., 'A1', 'Sheet1!B2')",
            placeholder="A1",
        )
        value: str = SchemaField(
            description="Value to write to the cell",
        )
        value_input_option: ValueInputOption = SchemaField(
            description="How input data should be interpreted",
            default=ValueInputOption.USER_ENTERED,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="The result of the update operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="df521b68-62d9-42e4-924f-fb6c245516fc",
            description="Update a single cell in a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsUpdateCellBlock.Input,
            output_schema=GoogleSheetsUpdateCellBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "cell": "A1",
                "value": "Hello World",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "result",
                    {"updatedCells": 1, "updatedColumns": 1, "updatedRows": 1},
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_update_cell": lambda *args, **kwargs: {
                    "updatedCells": 1,
                    "updatedColumns": 1,
                    "updatedRows": 1,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            if not input_data.spreadsheet:
                yield "error", "No spreadsheet selected"
                return

            # Check if the selected file is actually a Google Sheets spreadsheet
            validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
            if validation_error:
                yield "error", validation_error
                return

            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._update_cell,
                service,
                input_data.spreadsheet.id,
                input_data.cell,
                input_data.value,
                input_data.value_input_option,
            )

            yield "result", result
            # Output the GoogleDriveFile for chaining (preserves credentials_id)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", _handle_sheets_api_error(str(e), "update")

    def _update_cell(
        self,
        service,
        spreadsheet_id: str,
        cell: str,
        value: str,
        value_input_option: ValueInputOption,
    ) -> dict:
        body = {"values": [[value]]}
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=cell,
                valueInputOption=value_input_option.value,
                body=body,
            )
            .execute()
        )
        return {
            "updatedCells": result.get("updatedCells", 0),
            "updatedRows": result.get("updatedRows", 0),
            "updatedColumns": result.get("updatedColumns", 0),
        }


class FilterOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"


class SortOrder(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


def _column_letter_to_index(letter: str) -> int:
    """Convert column letter (A, B, ..., Z, AA, AB, ...) to 0-based index."""
    result = 0
    for char in letter.upper():
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _index_to_column_letter(index: int) -> str:
    """Convert 0-based column index to column letter (A, B, ..., Z, AA, AB, ...)."""
    result = ""
    index += 1  # Convert to 1-based
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _apply_filter(
    cell_value: str,
    filter_value: str,
    operator: FilterOperator,
    match_case: bool,
) -> bool:
    """Apply a filter condition to a cell value."""
    if operator == FilterOperator.IS_EMPTY:
        return cell_value.strip() == ""
    if operator == FilterOperator.IS_NOT_EMPTY:
        return cell_value.strip() != ""

    # For comparison operators, apply case sensitivity
    compare_cell = cell_value if match_case else cell_value.lower()
    compare_filter = filter_value if match_case else filter_value.lower()

    if operator == FilterOperator.EQUALS:
        return compare_cell == compare_filter
    elif operator == FilterOperator.NOT_EQUALS:
        return compare_cell != compare_filter
    elif operator == FilterOperator.CONTAINS:
        return compare_filter in compare_cell
    elif operator == FilterOperator.NOT_CONTAINS:
        return compare_filter not in compare_cell
    elif operator in (
        FilterOperator.GREATER_THAN,
        FilterOperator.LESS_THAN,
        FilterOperator.GREATER_THAN_OR_EQUAL,
        FilterOperator.LESS_THAN_OR_EQUAL,
    ):
        # Try numeric comparison first
        try:
            num_cell = float(cell_value)
            num_filter = float(filter_value)
            if operator == FilterOperator.GREATER_THAN:
                return num_cell > num_filter
            elif operator == FilterOperator.LESS_THAN:
                return num_cell < num_filter
            elif operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return num_cell >= num_filter
            elif operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return num_cell <= num_filter
        except ValueError:
            # Fall back to string comparison
            if operator == FilterOperator.GREATER_THAN:
                return compare_cell > compare_filter
            elif operator == FilterOperator.LESS_THAN:
                return compare_cell < compare_filter
            elif operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return compare_cell >= compare_filter
            elif operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return compare_cell <= compare_filter

    return False


class GoogleSheetsFilterRowsBlock(Block):
    """Filter rows in a Google Sheet based on column conditions."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        filter_column: str = SchemaField(
            description="Column to filter on (header name or column letter like 'A', 'B')",
            placeholder="Status",
        )
        filter_value: str = SchemaField(
            description="Value to filter by (not used for is_empty/is_not_empty operators)",
            default="",
        )
        operator: FilterOperator = SchemaField(
            description="Filter comparison operator",
            default=FilterOperator.EQUALS,
        )
        match_case: bool = SchemaField(
            description="Whether to match case in comparisons",
            default=False,
        )
        include_header: bool = SchemaField(
            description="Include header row in output",
            default=True,
        )

    class Output(BlockSchemaOutput):
        rows: list[list[str]] = SchemaField(
            description="Filtered rows (including header if requested)",
        )
        row_indices: list[int] = SchemaField(
            description="Original 1-based row indices of matching rows (useful for deletion)",
        )
        count: int = SchemaField(
            description="Number of matching rows (excluding header)",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="582195c2-ccee-4fc2-b646-18f72eb9906c",
            description="Filter rows in a Google Sheet based on a column condition. Returns matching rows and their indices.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsFilterRowsBlock.Input,
            output_schema=GoogleSheetsFilterRowsBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "filter_column": "Status",
                "filter_value": "Active",
                "operator": FilterOperator.EQUALS,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "rows",
                    [
                        ["Name", "Status", "Score"],
                        ["Alice", "Active", "85"],
                        ["Charlie", "Active", "92"],
                    ],
                ),
                ("row_indices", [2, 4]),
                ("count", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_filter_rows": lambda *args, **kwargs: {
                    "rows": [
                        ["Name", "Status", "Score"],
                        ["Alice", "Active", "85"],
                        ["Charlie", "Active", "92"],
                    ],
                    "row_indices": [2, 4],
                    "count": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._filter_rows,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.filter_column,
                input_data.filter_value,
                input_data.operator,
                input_data.match_case,
                input_data.include_header,
            )
            yield "rows", result["rows"]
            yield "row_indices", result["row_indices"]
            yield "count", result["count"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to filter rows: {str(e)}"

    def _filter_rows(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        filter_column: str,
        filter_value: str,
        operator: FilterOperator,
        match_case: bool,
        include_header: bool,
    ) -> dict:
        # Resolve sheet name
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        # Read all data from the sheet
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if not all_rows:
            return {"rows": [], "row_indices": [], "count": 0}

        header = all_rows[0]
        data_rows = all_rows[1:]

        # Determine filter column index
        filter_col_idx = -1

        # First, try to match against header names (handles "ID", "No", "To", etc.)
        for idx, col_name in enumerate(header):
            if (match_case and col_name == filter_column) or (
                not match_case and col_name.lower() == filter_column.lower()
            ):
                filter_col_idx = idx
                break

        # If no header match and looks like a column letter (A, B, AA, etc.), try that
        if filter_col_idx < 0 and filter_column.isalpha() and len(filter_column) <= 2:
            filter_col_idx = _column_letter_to_index(filter_column)
            # Validate column letter is within data range
            if filter_col_idx >= len(header):
                raise ValueError(
                    f"Column '{filter_column}' (index {filter_col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if filter_col_idx < 0:
            raise ValueError(
                f"Column '{filter_column}' not found. Available columns: {header}"
            )

        # Filter rows
        filtered_rows = []
        row_indices = []

        for row_idx, row in enumerate(data_rows):
            # Get cell value (handle rows shorter than filter column)
            cell_value = row[filter_col_idx] if filter_col_idx < len(row) else ""

            if _apply_filter(str(cell_value), filter_value, operator, match_case):
                filtered_rows.append(row)
                row_indices.append(row_idx + 2)  # +2 for 1-based index and header

        # Prepare output
        output_rows = []
        if include_header:
            output_rows.append(header)
        output_rows.extend(filtered_rows)

        return {
            "rows": output_rows,
            "row_indices": row_indices,
            "count": len(filtered_rows),
        }


class GoogleSheetsLookupRowBlock(Block):
    """Look up a row by matching a value in a column (VLOOKUP-style)."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        lookup_column: str = SchemaField(
            description="Column to search in (header name or column letter)",
            placeholder="ID",
        )
        lookup_value: str = SchemaField(
            description="Value to search for",
        )
        return_columns: list[str] = SchemaField(
            description="Columns to return (header names or letters). Empty = all columns.",
            default=[],
        )
        match_case: bool = SchemaField(
            description="Whether to match case",
            default=False,
        )

    class Output(BlockSchemaOutput):
        row: list[str] = SchemaField(
            description="The matching row (all or selected columns)",
        )
        row_dict: dict[str, str] = SchemaField(
            description="The matching row as a dictionary (header: value)",
        )
        row_index: int = SchemaField(
            description="1-based row index of the match",
        )
        found: bool = SchemaField(
            description="Whether a match was found",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="e58c0bad-6597-400c-9548-d151ec428ffc",
            description="Look up a row by finding a value in a specific column. Returns the first matching row and optionally specific columns.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsLookupRowBlock.Input,
            output_schema=GoogleSheetsLookupRowBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "lookup_column": "ID",
                "lookup_value": "123",
                "return_columns": ["Name", "Email"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("row", ["Alice", "alice@example.com"]),
                ("row_dict", {"Name": "Alice", "Email": "alice@example.com"}),
                ("row_index", 2),
                ("found", True),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_lookup_row": lambda *args, **kwargs: {
                    "row": ["Alice", "alice@example.com"],
                    "row_dict": {"Name": "Alice", "Email": "alice@example.com"},
                    "row_index": 2,
                    "found": True,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._lookup_row,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.lookup_column,
                input_data.lookup_value,
                input_data.return_columns,
                input_data.match_case,
            )
            yield "row", result["row"]
            yield "row_dict", result["row_dict"]
            yield "row_index", result["row_index"]
            yield "found", result["found"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to lookup row: {str(e)}"

    def _lookup_row(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        lookup_column: str,
        lookup_value: str,
        return_columns: list[str],
        match_case: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if not all_rows:
            return {"row": [], "row_dict": {}, "row_index": 0, "found": False}

        header = all_rows[0]
        data_rows = all_rows[1:]

        # Find lookup column index - first try header name match, then column letter
        lookup_col_idx = -1
        for idx, col_name in enumerate(header):
            if (match_case and col_name == lookup_column) or (
                not match_case and col_name.lower() == lookup_column.lower()
            ):
                lookup_col_idx = idx
                break

        # If no header match and looks like a column letter, try that
        if lookup_col_idx < 0 and lookup_column.isalpha() and len(lookup_column) <= 2:
            lookup_col_idx = _column_letter_to_index(lookup_column)
            # Validate column letter is within data range
            if lookup_col_idx >= len(header):
                raise ValueError(
                    f"Column '{lookup_column}' (index {lookup_col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if lookup_col_idx < 0:
            raise ValueError(
                f"Lookup column '{lookup_column}' not found. Available: {header}"
            )

        # Find return column indices - first try header name match, then column letter
        return_col_indices = []
        return_col_headers = []
        if return_columns:
            for ret_col in return_columns:
                found = False
                # First try header name match
                for idx, col_name in enumerate(header):
                    if (match_case and col_name == ret_col) or (
                        not match_case and col_name.lower() == ret_col.lower()
                    ):
                        return_col_indices.append(idx)
                        return_col_headers.append(col_name)
                        found = True
                        break

                # If no header match and looks like a column letter, try that
                if not found and ret_col.isalpha() and len(ret_col) <= 2:
                    idx = _column_letter_to_index(ret_col)
                    # Validate column letter is within data range
                    if idx >= len(header):
                        raise ValueError(
                            f"Return column '{ret_col}' (index {idx}) is out of range. "
                            f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                        )
                    return_col_indices.append(idx)
                    return_col_headers.append(header[idx])
                    found = True

                if not found:
                    raise ValueError(
                        f"Return column '{ret_col}' not found. Available: {header}"
                    )
        else:
            return_col_indices = list(range(len(header)))
            return_col_headers = header

        # Search for matching row
        compare_value = lookup_value if match_case else lookup_value.lower()

        for row_idx, row in enumerate(data_rows):
            cell_value = row[lookup_col_idx] if lookup_col_idx < len(row) else ""
            compare_cell = str(cell_value) if match_case else str(cell_value).lower()

            if compare_cell == compare_value:
                # Found a match - extract requested columns
                result_row = []
                result_dict = {}
                for i, col_idx in enumerate(return_col_indices):
                    value = row[col_idx] if col_idx < len(row) else ""
                    result_row.append(value)
                    result_dict[return_col_headers[i]] = value

                return {
                    "row": result_row,
                    "row_dict": result_dict,
                    "row_index": row_idx + 2,
                    "found": True,
                }

        return {"row": [], "row_dict": {}, "row_index": 0, "found": False}


class GoogleSheetsDeleteRowsBlock(Block):
    """Delete rows from a Google Sheet by row indices."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        row_indices: list[int] = SchemaField(
            description="1-based row indices to delete (e.g., [2, 5, 7])",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the delete operation",
        )
        deleted_count: int = SchemaField(
            description="Number of rows deleted",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="24bcd490-b02d-44c6-847d-b62a2319f5eb",
            description="Delete specific rows from a Google Sheet by their row indices. Works well with FilterRowsBlock output.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsDeleteRowsBlock.Input,
            output_schema=GoogleSheetsDeleteRowsBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "row_indices": [2, 5],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("deleted_count", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_delete_rows": lambda *args, **kwargs: {
                    "success": True,
                    "deleted_count": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._delete_rows,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.row_indices,
            )
            yield "result", {"success": True}
            yield "deleted_count", result["deleted_count"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to delete rows: {str(e)}"

    def _delete_rows(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        row_indices: list[int],
    ) -> dict:
        if not row_indices:
            return {"success": True, "deleted_count": 0}

        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Deduplicate and sort row indices in descending order to delete from bottom to top
        # Deduplication prevents deleting wrong rows if same index appears multiple times
        sorted_indices = sorted(set(row_indices), reverse=True)

        # Build delete requests
        requests = []
        for row_idx in sorted_indices:
            # Convert to 0-based index
            start_idx = row_idx - 1
            requests.append(
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": start_idx,
                            "endIndex": start_idx + 1,
                        }
                    }
                }
            )

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()

        return {"success": True, "deleted_count": len(sorted_indices)}


class GoogleSheetsGetColumnBlock(Block):
    """Get all values from a specific column by header name."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        column: str = SchemaField(
            description="Column to extract (header name or column letter like 'A', 'B')",
            placeholder="Email",
        )
        include_header: bool = SchemaField(
            description="Include the header in output",
            default=False,
        )
        skip_empty: bool = SchemaField(
            description="Skip empty cells",
            default=False,
        )

    class Output(BlockSchemaOutput):
        values: list[str] = SchemaField(
            description="List of values from the column",
        )
        count: int = SchemaField(
            description="Number of values (excluding header if not included)",
        )
        column_index: int = SchemaField(
            description="0-based column index",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="108d911f-e109-47fb-addc-2259792ee850",
            description="Extract all values from a specific column. Useful for getting a list of emails, IDs, or any single field.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsGetColumnBlock.Input,
            output_schema=GoogleSheetsGetColumnBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "column": "Email",
                "include_header": False,
                "skip_empty": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "values",
                    ["alice@example.com", "bob@example.com", "charlie@example.com"],
                ),
                ("count", 3),
                ("column_index", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_column": lambda *args, **kwargs: {
                    "values": [
                        "alice@example.com",
                        "bob@example.com",
                        "charlie@example.com",
                    ],
                    "count": 3,
                    "column_index": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_column,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.column,
                input_data.include_header,
                input_data.skip_empty,
            )
            yield "values", result["values"]
            yield "count", result["count"]
            yield "column_index", result["column_index"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get column: {str(e)}"

    def _get_column(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        column: str,
        include_header: bool,
        skip_empty: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if not all_rows:
            return {"values": [], "count": 0, "column_index": -1}

        header = all_rows[0]

        # Find column index - first try header name match, then column letter
        col_idx = -1
        for idx, col_name in enumerate(header):
            if col_name.lower() == column.lower():
                col_idx = idx
                break

        # If no header match and looks like a column letter, try that
        if col_idx < 0 and column.isalpha() and len(column) <= 2:
            col_idx = _column_letter_to_index(column)
            # Validate column letter is within data range
            if col_idx >= len(header):
                raise ValueError(
                    f"Column '{column}' (index {col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if col_idx < 0:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {header}"
            )

        # Extract column values
        values = []
        start_row = 0 if include_header else 1

        for row in all_rows[start_row:]:
            value = row[col_idx] if col_idx < len(row) else ""
            if skip_empty and not str(value).strip():
                continue
            values.append(str(value))

        return {"values": values, "count": len(values), "column_index": col_idx}


class GoogleSheetsSortBlock(Block):
    """Sort a Google Sheet by one or more columns."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        sort_column: str = SchemaField(
            description="Primary column to sort by (header name or column letter)",
            placeholder="Date",
        )
        sort_order: SortOrder = SchemaField(
            description="Sort order for primary column",
            default=SortOrder.ASCENDING,
        )
        secondary_column: str = SchemaField(
            description="Secondary column to sort by (optional)",
            default="",
        )
        secondary_order: SortOrder = SchemaField(
            description="Sort order for secondary column",
            default=SortOrder.ASCENDING,
        )
        has_header: bool = SchemaField(
            description="Whether the data has a header row (header won't be sorted)",
            default=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the sort operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="a265bd84-c93b-459d-bbe0-94e6addaa38f",
            description="Sort a Google Sheet by one or two columns. The sheet is sorted in-place.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsSortBlock.Input,
            output_schema=GoogleSheetsSortBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "sort_column": "Score",
                "sort_order": SortOrder.DESCENDING,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_sort_sheet": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._sort_sheet,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.sort_column,
                input_data.sort_order,
                input_data.secondary_column,
                input_data.secondary_order,
                input_data.has_header,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to sort sheet: {str(e)}"

    def _sort_sheet(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        sort_column: str,
        sort_order: SortOrder,
        secondary_column: str,
        secondary_order: SortOrder,
        has_header: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Get sheet metadata to find column indices and grid properties
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_meta = None
        for sheet in meta.get("sheets", []):
            if sheet.get("properties", {}).get("sheetId") == sheet_id:
                sheet_meta = sheet
                break

        if not sheet_meta:
            raise ValueError(f"Could not find metadata for sheet '{target_sheet}'")

        grid_props = sheet_meta.get("properties", {}).get("gridProperties", {})
        row_count = grid_props.get("rowCount", 1000)
        col_count = grid_props.get("columnCount", 26)

        # Get header to resolve column names
        formatted_sheet = format_sheet_name(target_sheet)
        header_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"{formatted_sheet}!1:1")
            .execute()
        )
        header = (
            header_result.get("values", [[]])[0] if header_result.get("values") else []
        )

        # Find primary sort column index - first try header name match, then column letter
        sort_col_idx = -1
        for idx, col_name in enumerate(header):
            if col_name.lower() == sort_column.lower():
                sort_col_idx = idx
                break

        # If no header match and looks like a column letter, try that
        if sort_col_idx < 0 and sort_column.isalpha() and len(sort_column) <= 2:
            sort_col_idx = _column_letter_to_index(sort_column)
            # Validate column letter is within data range
            if sort_col_idx >= len(header):
                raise ValueError(
                    f"Sort column '{sort_column}' (index {sort_col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if sort_col_idx < 0:
            raise ValueError(
                f"Sort column '{sort_column}' not found. Available: {header}"
            )

        # Build sort specs
        sort_specs = [
            {
                "dimensionIndex": sort_col_idx,
                "sortOrder": (
                    "ASCENDING" if sort_order == SortOrder.ASCENDING else "DESCENDING"
                ),
            }
        ]

        # Add secondary sort if specified
        if secondary_column:
            sec_col_idx = -1
            # First try header name match
            for idx, col_name in enumerate(header):
                if col_name.lower() == secondary_column.lower():
                    sec_col_idx = idx
                    break

            # If no header match and looks like a column letter, try that
            if (
                sec_col_idx < 0
                and secondary_column.isalpha()
                and len(secondary_column) <= 2
            ):
                sec_col_idx = _column_letter_to_index(secondary_column)
                # Validate column letter is within data range
                if sec_col_idx >= len(header):
                    raise ValueError(
                        f"Secondary sort column '{secondary_column}' (index {sec_col_idx}) is out of range. "
                        f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                    )

            if sec_col_idx < 0:
                raise ValueError(
                    f"Secondary sort column '{secondary_column}' not found. Available: {header}"
                )

            sort_specs.append(
                {
                    "dimensionIndex": sec_col_idx,
                    "sortOrder": (
                        "ASCENDING"
                        if secondary_order == SortOrder.ASCENDING
                        else "DESCENDING"
                    ),
                }
            )

        # Build sort range request
        start_row = 1 if has_header else 0  # Skip header if present

        request = {
            "sortRange": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": row_count,
                    "startColumnIndex": 0,
                    "endColumnIndex": col_count,
                },
                "sortSpecs": sort_specs,
            }
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [request]}
        ).execute()

        return {"success": True}


class GoogleSheetsGetUniqueValuesBlock(Block):
    """Get unique values from a column."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        column: str = SchemaField(
            description="Column to get unique values from (header name or column letter)",
            placeholder="Category",
        )
        include_count: bool = SchemaField(
            description="Include count of each unique value",
            default=False,
        )
        sort_by_count: bool = SchemaField(
            description="Sort results by count (most frequent first)",
            default=False,
        )

    class Output(BlockSchemaOutput):
        values: list[str] = SchemaField(
            description="List of unique values",
        )
        counts: dict[str, int] = SchemaField(
            description="Count of each unique value (if include_count is True)",
        )
        total_unique: int = SchemaField(
            description="Total number of unique values",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="0f296c0b-6b6e-4280-b96e-ae1459b98dff",
            description="Get unique values from a column. Useful for building dropdown options or finding distinct categories.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsGetUniqueValuesBlock.Input,
            output_schema=GoogleSheetsGetUniqueValuesBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "column": "Status",
                "include_count": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("values", ["Active", "Inactive", "Pending"]),
                ("counts", {"Active": 5, "Inactive": 3, "Pending": 2}),
                ("total_unique", 3),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_unique_values": lambda *args, **kwargs: {
                    "values": ["Active", "Inactive", "Pending"],
                    "counts": {"Active": 5, "Inactive": 3, "Pending": 2},
                    "total_unique": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_unique_values,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.column,
                input_data.include_count,
                input_data.sort_by_count,
            )
            yield "values", result["values"]
            yield "counts", result["counts"]
            yield "total_unique", result["total_unique"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get unique values: {str(e)}"

    def _get_unique_values(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        column: str,
        include_count: bool,
        sort_by_count: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if not all_rows:
            return {"values": [], "counts": {}, "total_unique": 0}

        header = all_rows[0]

        # Find column index - first try header name match, then column letter
        col_idx = -1
        for idx, col_name in enumerate(header):
            if col_name.lower() == column.lower():
                col_idx = idx
                break

        # If no header match and looks like a column letter, try that
        if col_idx < 0 and column.isalpha() and len(column) <= 2:
            col_idx = _column_letter_to_index(column)
            # Validate column letter is within data range
            if col_idx >= len(header):
                raise ValueError(
                    f"Column '{column}' (index {col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if col_idx < 0:
            raise ValueError(
                f"Column '{column}' not found. Available columns: {header}"
            )

        # Count values
        value_counts: dict[str, int] = {}
        for row in all_rows[1:]:  # Skip header
            value = str(row[col_idx]) if col_idx < len(row) else ""
            if value.strip():  # Skip empty values
                value_counts[value] = value_counts.get(value, 0) + 1

        # Sort values
        if sort_by_count:
            sorted_items = sorted(value_counts.items(), key=lambda x: -x[1])
            unique_values = [item[0] for item in sorted_items]
        else:
            unique_values = sorted(value_counts.keys())

        return {
            "values": unique_values,
            "counts": value_counts if include_count else {},
            "total_unique": len(unique_values),
        }


class GoogleSheetsInsertRowBlock(Block):
    """Insert a single row at a specific position in a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        row: list[str] = SchemaField(
            description="Row values to insert (e.g., ['Alice', 'alice@example.com', '25'])",
        )
        row_index: int = SchemaField(
            description="1-based row index where to insert (existing rows shift down)",
            placeholder="2",
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        value_input_option: ValueInputOption = SchemaField(
            description="How values are interpreted. USER_ENTERED: parsed like typed input (e.g., '=SUM(A1:A5)' becomes a formula, '1/2/2024' becomes a date). RAW: stored as-is without parsing.",
            default=ValueInputOption.USER_ENTERED,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the insert operation")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="03eda5df-8080-4ed1-bfdf-212f543d657e",
            description="Insert a single row at a specific position. Existing rows shift down.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsInsertRowBlock.Input,
            output_schema=GoogleSheetsInsertRowBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "row": ["New", "Row", "Data"],
                "row_index": 3,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_insert_row": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.row:
            yield "error", "Row data is required"
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._insert_row,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.row_index,
                input_data.row,
                input_data.value_input_option,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to insert row: {str(e)}"

    def _insert_row(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        row_index: int,
        row: list[str],
        value_input_option: ValueInputOption,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        start_idx = row_index - 1  # Convert to 0-based

        # First, insert an empty row
        insert_request = {
            "insertDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": start_idx,
                    "endIndex": start_idx + 1,
                },
                "inheritFromBefore": start_idx > 0,
            }
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [insert_request]}
        ).execute()

        # Then, write the values
        formatted_sheet = format_sheet_name(target_sheet)
        write_range = f"{formatted_sheet}!A{row_index}"

        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=write_range,
            valueInputOption=value_input_option.value,
            body={"values": [row]},  # Wrap single row in list for API
        ).execute()

        return {"success": True}


class GoogleSheetsAddColumnBlock(Block):
    """Add a new column with a header to a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        header: str = SchemaField(
            description="Header name for the new column",
            placeholder="New Column",
        )
        position: str = SchemaField(
            description="Where to add: 'end' for last column, or column letter (e.g., 'C') to insert before",
            default="end",
        )
        default_value: str = SchemaField(
            description="Default value to fill in all data rows (optional). Requires existing data rows.",
            default="",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the operation",
        )
        column_letter: str = SchemaField(
            description="Letter of the new column (e.g., 'D')",
        )
        column_index: int = SchemaField(
            description="0-based index of the new column",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="cac51050-fc9e-4e63-987a-66c2ba2a127b",
            description="Add a new column with a header. Can add at the end or insert at a specific position.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAddColumnBlock.Input,
            output_schema=GoogleSheetsAddColumnBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "header": "New Status",
                "position": "end",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("column_letter", "D"),
                ("column_index", 3),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_add_column": lambda *args, **kwargs: {
                    "success": True,
                    "column_letter": "D",
                    "column_index": 3,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._add_column,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.header,
                input_data.position,
                input_data.default_value,
            )
            yield "result", {"success": True}
            yield "column_letter", result["column_letter"]
            yield "column_index", result["column_index"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to add column: {str(e)}"

    def _add_column(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        header: str,
        position: str,
        default_value: str,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Get current data to determine column count and row count
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])
        current_col_count = max(len(row) for row in all_rows) if all_rows else 0
        row_count = len(all_rows)

        # Determine target column index
        if position.lower() == "end":
            col_idx = current_col_count
        elif position.isalpha() and len(position) <= 2:
            col_idx = _column_letter_to_index(position)
            # Insert a new column at this position
            insert_request = {
                "insertDimension": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": col_idx,
                        "endIndex": col_idx + 1,
                    },
                    "inheritFromBefore": col_idx > 0,
                }
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": [insert_request]}
            ).execute()
        else:
            raise ValueError(
                f"Invalid position: '{position}'. Use 'end' or a column letter."
            )

        col_letter = _index_to_column_letter(col_idx)

        # Write header
        header_range = f"{formatted_sheet}!{col_letter}1"
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=header_range,
            valueInputOption="USER_ENTERED",
            body={"values": [[header]]},
        ).execute()

        # Fill default value if provided and there are data rows
        if default_value and row_count > 1:
            values_to_fill = [[default_value]] * (row_count - 1)
            data_range = f"{formatted_sheet}!{col_letter}2:{col_letter}{row_count}"
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=data_range,
                valueInputOption="USER_ENTERED",
                body={"values": values_to_fill},
            ).execute()

        return {
            "success": True,
            "column_letter": col_letter,
            "column_index": col_idx,
        }


class GoogleSheetsGetRowCountBlock(Block):
    """Get the number of rows in a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        include_header: bool = SchemaField(
            description="Include header row in count",
            default=True,
        )
        count_empty: bool = SchemaField(
            description="Count rows with only empty cells",
            default=False,
        )

    class Output(BlockSchemaOutput):
        total_rows: int = SchemaField(
            description="Total number of rows",
        )
        data_rows: int = SchemaField(
            description="Number of data rows (excluding header)",
        )
        last_row: int = SchemaField(
            description="1-based index of the last row with data",
        )
        column_count: int = SchemaField(
            description="Number of columns",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="080cc84b-a94a-4fb4-90e3-dcc55ee783af",
            description="Get row count and dimensions of a Google Sheet. Useful for knowing where data ends.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsGetRowCountBlock.Input,
            output_schema=GoogleSheetsGetRowCountBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("total_rows", 101),
                ("data_rows", 100),
                ("last_row", 101),
                ("column_count", 5),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_row_count": lambda *args, **kwargs: {
                    "total_rows": 101,
                    "data_rows": 100,
                    "last_row": 101,
                    "column_count": 5,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_row_count,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.include_header,
                input_data.count_empty,
            )
            yield "total_rows", result["total_rows"]
            yield "data_rows", result["data_rows"]
            yield "last_row", result["last_row"]
            yield "column_count", result["column_count"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get row count: {str(e)}"

    def _get_row_count(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        include_header: bool,
        count_empty: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if not all_rows:
            return {
                "total_rows": 0,
                "data_rows": 0,
                "last_row": 0,
                "column_count": 0,
            }

        # Count non-empty rows
        if count_empty:
            total_rows = len(all_rows)
            last_row = total_rows
        else:
            # Find last row with actual data
            last_row = 0
            for idx, row in enumerate(all_rows):
                if any(str(cell).strip() for cell in row):
                    last_row = idx + 1
            total_rows = last_row

        data_rows = total_rows - 1 if total_rows > 0 else 0
        if not include_header:
            total_rows = data_rows

        column_count = max(len(row) for row in all_rows) if all_rows else 0

        return {
            "total_rows": total_rows,
            "data_rows": data_rows,
            "last_row": last_row,
            "column_count": column_count,
        }


class GoogleSheetsRemoveDuplicatesBlock(Block):
    """Remove duplicate rows from a Google Sheet based on specified columns."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        columns: list[str] = SchemaField(
            description="Columns to check for duplicates (header names or letters). Empty = all columns.",
            default=[],
        )
        keep: str = SchemaField(
            description="Which duplicate to keep: 'first' or 'last'",
            default="first",
        )
        match_case: bool = SchemaField(
            description="Whether to match case when comparing",
            default=False,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the operation",
        )
        removed_count: int = SchemaField(
            description="Number of duplicate rows removed",
        )
        remaining_rows: int = SchemaField(
            description="Number of rows remaining",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="6eb50ff7-205b-400e-8ecc-1ce8d50075be",
            description="Remove duplicate rows based on specified columns. Keeps either the first or last occurrence.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsRemoveDuplicatesBlock.Input,
            output_schema=GoogleSheetsRemoveDuplicatesBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "columns": ["Email"],
                "keep": "first",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("removed_count", 5),
                ("remaining_rows", 95),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_remove_duplicates": lambda *args, **kwargs: {
                    "success": True,
                    "removed_count": 5,
                    "remaining_rows": 95,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._remove_duplicates,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.columns,
                input_data.keep,
                input_data.match_case,
            )
            yield "result", {"success": True}
            yield "removed_count", result["removed_count"]
            yield "remaining_rows", result["remaining_rows"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to remove duplicates: {str(e)}"

    def _remove_duplicates(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        columns: list[str],
        keep: str,
        match_case: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Read all data
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=formatted_sheet)
            .execute()
        )
        all_rows = result.get("values", [])

        if len(all_rows) <= 1:  # Only header or empty
            return {
                "success": True,
                "removed_count": 0,
                "remaining_rows": len(all_rows),
            }

        header = all_rows[0]
        data_rows = all_rows[1:]

        # Determine which column indices to use for comparison
        # First try header name match, then column letter
        if columns:
            col_indices = []
            for col in columns:
                found = False
                # First try header name match
                for idx, col_name in enumerate(header):
                    if col_name.lower() == col.lower():
                        col_indices.append(idx)
                        found = True
                        break

                # If no header match and looks like a column letter, try that
                if not found and col.isalpha() and len(col) <= 2:
                    col_idx = _column_letter_to_index(col)
                    # Validate column letter is within data range
                    if col_idx >= len(header):
                        raise ValueError(
                            f"Column '{col}' (index {col_idx}) is out of range. "
                            f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                        )
                    col_indices.append(col_idx)
                    found = True

                if not found:
                    raise ValueError(
                        f"Column '{col}' not found in sheet. "
                        f"Available columns: {', '.join(header)}"
                    )
        else:
            col_indices = list(range(len(header)))

        # Find duplicates
        seen: dict[tuple, int] = {}
        rows_to_delete: list[int] = []

        for row_idx, row in enumerate(data_rows):
            # Build key from specified columns
            key_parts = []
            for col_idx in col_indices:
                value = str(row[col_idx]) if col_idx < len(row) else ""
                if not match_case:
                    value = value.lower()
                key_parts.append(value)
            key = tuple(key_parts)

            if key in seen:
                if keep == "first":
                    # Delete this row (keep the first one we saw)
                    rows_to_delete.append(row_idx + 2)  # +2 for 1-based and header
                else:
                    # Delete the previous row, then update seen to keep this one
                    prev_row = seen[key]
                    rows_to_delete.append(prev_row)
                    seen[key] = row_idx + 2
            else:
                seen[key] = row_idx + 2

        if not rows_to_delete:
            return {
                "success": True,
                "removed_count": 0,
                "remaining_rows": len(all_rows),
            }

        # Sort in descending order to delete from bottom to top
        rows_to_delete = sorted(set(rows_to_delete), reverse=True)

        # Delete rows
        requests = []
        for row_idx in rows_to_delete:
            start_idx = row_idx - 1
            requests.append(
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": start_idx,
                            "endIndex": start_idx + 1,
                        }
                    }
                }
            )

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()

        remaining = len(all_rows) - len(rows_to_delete)
        return {
            "success": True,
            "removed_count": len(rows_to_delete),
            "remaining_rows": remaining,
        }


class GoogleSheetsUpdateRowBlock(Block):
    """Update a specific row by index with new values."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        row_index: int = SchemaField(
            description="1-based row index to update",
        )
        values: list[str] = SchemaField(
            description="New values for the row (in column order)",
            default=[],
        )
        dict_values: dict[str, str] = SchemaField(
            description="Values as dict with column headers as keys (alternative to values)",
            default={},
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the update operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="b8a934d5-fca0-4be3-9fc2-a99bf63bd385",
            description="Update a specific row by its index. Can use list or dict format for values.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsUpdateRowBlock.Input,
            output_schema=GoogleSheetsUpdateRowBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "row_index": 5,
                "dict_values": {"Name": "Updated Name", "Status": "Active"},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "updatedCells": 2}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_update_row": lambda *args, **kwargs: {
                    "success": True,
                    "updatedCells": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.values and not input_data.dict_values:
            yield "error", "Either values or dict_values must be provided"
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._update_row,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.row_index,
                input_data.values,
                input_data.dict_values,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to update row: {str(e)}"

    def _update_row(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        row_index: int,
        values: list[str],
        dict_values: dict[str, str],
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        if dict_values:
            # Get header to map column names to indices
            header_result = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=f"{formatted_sheet}!1:1")
                .execute()
            )
            header = (
                header_result.get("values", [[]])[0]
                if header_result.get("values")
                else []
            )

            # Get current row values
            row_range = f"{formatted_sheet}!{row_index}:{row_index}"
            current_result = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=row_range)
                .execute()
            )
            current_row = (
                current_result.get("values", [[]])[0]
                if current_result.get("values")
                else []
            )

            # Extend current row to match header length
            while len(current_row) < len(header):
                current_row.append("")

            # Update specific columns from dict - validate all column names first
            for col_name in dict_values.keys():
                found = False
                for h in header:
                    if h.lower() == col_name.lower():
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"Column '{col_name}' not found in sheet. "
                        f"Available columns: {', '.join(header)}"
                    )

            # Now apply updates
            updated_count = 0
            for col_name, value in dict_values.items():
                for idx, h in enumerate(header):
                    if h.lower() == col_name.lower():
                        current_row[idx] = value
                        updated_count += 1
                        break

            values = current_row
        else:
            updated_count = len(values)

        # Write the row
        write_range = f"{formatted_sheet}!A{row_index}"
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=write_range,
            valueInputOption="USER_ENTERED",
            body={"values": [values]},
        ).execute()

        return {"success": True, "updatedCells": updated_count}


class GoogleSheetsGetRowBlock(Block):
    """Get a specific row by its index."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        row_index: int = SchemaField(
            description="1-based row index to retrieve",
        )

    class Output(BlockSchemaOutput):
        row: list[str] = SchemaField(
            description="The row values as a list",
        )
        row_dict: dict[str, str] = SchemaField(
            description="The row as a dictionary (header: value)",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="c4be9390-2431-4682-9769-7025b22a5fa7",
            description="Get a specific row by its index. Returns both list and dict formats.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsGetRowBlock.Input,
            output_schema=GoogleSheetsGetRowBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "row_index": 3,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("row", ["Alice", "Active", "85"]),
                ("row_dict", {"Name": "Alice", "Status": "Active", "Score": "85"}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_row": lambda *args, **kwargs: {
                    "row": ["Alice", "Active", "85"],
                    "row_dict": {"Name": "Alice", "Status": "Active", "Score": "85"},
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_row,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.row_index,
            )
            yield "row", result["row"]
            yield "row_dict", result["row_dict"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get row: {str(e)}"

    def _get_row(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        row_index: int,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)

        # Get header
        header_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"{formatted_sheet}!1:1")
            .execute()
        )
        header = (
            header_result.get("values", [[]])[0] if header_result.get("values") else []
        )

        # Get the row
        row_range = f"{formatted_sheet}!{row_index}:{row_index}"
        row_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=row_range)
            .execute()
        )
        row = row_result.get("values", [[]])[0] if row_result.get("values") else []

        # Build dictionary
        row_dict = {}
        for idx, h in enumerate(header):
            row_dict[h] = row[idx] if idx < len(row) else ""

        return {"row": row, "row_dict": row_dict}


class GoogleSheetsDeleteColumnBlock(Block):
    """Delete a column from a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        column: str = SchemaField(
            description="Column to delete (header name or column letter like 'A', 'B')",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the delete operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="59b266b6-5cce-4661-a1d3-c417e64d68e9",
            description="Delete a column by header name or column letter.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsDeleteColumnBlock.Input,
            output_schema=GoogleSheetsDeleteColumnBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "column": "Status",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_delete_column": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._delete_column,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.column,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to delete column: {str(e)}"

    def _delete_column(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        column: str,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        formatted_sheet = format_sheet_name(target_sheet)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Get header to find column by name or validate column letter
        header_result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"{formatted_sheet}!1:1")
            .execute()
        )
        header = (
            header_result.get("values", [[]])[0] if header_result.get("values") else []
        )

        # Find column index - first try header name match, then column letter
        col_idx = -1
        for idx, h in enumerate(header):
            if h.lower() == column.lower():
                col_idx = idx
                break

        # If no header match and looks like a column letter, try that
        if col_idx < 0 and column.isalpha() and len(column) <= 2:
            col_idx = _column_letter_to_index(column)
            # Validate column letter is within data range
            if col_idx >= len(header):
                raise ValueError(
                    f"Column '{column}' (index {col_idx}) is out of range. "
                    f"Sheet only has {len(header)} columns (A-{_index_to_column_letter(len(header) - 1)})."
                )

        if col_idx < 0:
            raise ValueError(f"Column '{column}' not found")

        # Delete the column
        request = {
            "deleteDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": col_idx,
                    "endIndex": col_idx + 1,
                }
            }
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [request]}
        ).execute()

        return {"success": True}


class GoogleSheetsCreateNamedRangeBlock(Block):
    """Create a named range in a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        name: str = SchemaField(
            description="Name for the range (e.g., 'SalesData', 'CustomerList')",
            placeholder="MyNamedRange",
        )
        range: str = SchemaField(
            description="Cell range in A1 notation (e.g., 'A1:D10', 'B2:B100')",
            placeholder="A1:D10",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the operation",
        )
        named_range_id: str = SchemaField(
            description="ID of the created named range",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="a2707376-8016-494b-98c4-d0e2752ab9cb",
            description="Create a named range to reference cells by name instead of A1 notation.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsCreateNamedRangeBlock.Input,
            output_schema=GoogleSheetsCreateNamedRangeBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "name": "SalesData",
                "range": "A1:D10",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("named_range_id", "nr_12345"),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_create_named_range": lambda *args, **kwargs: {
                    "success": True,
                    "named_range_id": "nr_12345",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._create_named_range,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.name,
                input_data.range,
            )
            yield "result", {"success": True}
            yield "named_range_id", result["named_range_id"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to create named range: {str(e)}"

    def _create_named_range(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        name: str,
        range_str: str,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Parse range to get grid coordinates
        # Handle both "A1:D10" and "Sheet1!A1:D10" formats
        if "!" in range_str:
            range_str = range_str.split("!")[1]

        # Parse start and end cells
        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_str.upper())
        if not match:
            raise ValueError(f"Invalid range format: {range_str}")

        start_col = _column_letter_to_index(match.group(1))
        start_row = int(match.group(2)) - 1  # 0-based
        end_col = _column_letter_to_index(match.group(3)) + 1  # exclusive
        end_row = int(match.group(4))  # exclusive (already 1-based becomes 0-based + 1)

        request = {
            "addNamedRange": {
                "namedRange": {
                    "name": name,
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row,
                        "endRowIndex": end_row,
                        "startColumnIndex": start_col,
                        "endColumnIndex": end_col,
                    },
                }
            }
        }

        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": [request]})
            .execute()
        )

        # Extract the named range ID from the response
        named_range_id = ""
        replies = result.get("replies", [])
        if replies and "addNamedRange" in replies[0]:
            named_range_id = replies[0]["addNamedRange"]["namedRange"]["namedRangeId"]

        return {"success": True, "named_range_id": named_range_id}


class GoogleSheetsListNamedRangesBlock(Block):
    """List all named ranges in a Google Sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )

    class Output(BlockSchemaOutput):
        named_ranges: list[dict] = SchemaField(
            description="List of named ranges with name, id, and range info",
        )
        count: int = SchemaField(
            description="Number of named ranges",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="b81a9d27-3997-4860-9303-cc68086db13a",
            description="List all named ranges in a spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsListNamedRangesBlock.Input,
            output_schema=GoogleSheetsListNamedRangesBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "named_ranges",
                    [
                        {"name": "SalesData", "id": "nr_1", "range": "Sheet1!A1:D10"},
                        {
                            "name": "CustomerList",
                            "id": "nr_2",
                            "range": "Sheet1!E1:F50",
                        },
                    ],
                ),
                ("count", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_list_named_ranges": lambda *args, **kwargs: {
                    "named_ranges": [
                        {"name": "SalesData", "id": "nr_1", "range": "Sheet1!A1:D10"},
                        {
                            "name": "CustomerList",
                            "id": "nr_2",
                            "range": "Sheet1!E1:F50",
                        },
                    ],
                    "count": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._list_named_ranges,
                service,
                input_data.spreadsheet.id,
            )
            yield "named_ranges", result["named_ranges"]
            yield "count", result["count"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to list named ranges: {str(e)}"

    def _list_named_ranges(
        self,
        service,
        spreadsheet_id: str,
    ) -> dict:
        # Get spreadsheet metadata including named ranges
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

        named_ranges_list = []
        named_ranges = meta.get("namedRanges", [])

        # Get sheet names for reference
        sheets = {
            sheet["properties"]["sheetId"]: sheet["properties"]["title"]
            for sheet in meta.get("sheets", [])
        }

        for nr in named_ranges:
            range_info = nr.get("range", {})
            sheet_id = range_info.get("sheetId", 0)
            sheet_name = sheets.get(sheet_id, "Sheet1")

            # Convert grid range back to A1 notation
            start_col = _index_to_column_letter(range_info.get("startColumnIndex", 0))
            end_col = _index_to_column_letter(range_info.get("endColumnIndex", 1) - 1)
            start_row = range_info.get("startRowIndex", 0) + 1
            end_row = range_info.get("endRowIndex", 1)

            range_str = f"{sheet_name}!{start_col}{start_row}:{end_col}{end_row}"

            named_ranges_list.append(
                {
                    "name": nr.get("name", ""),
                    "id": nr.get("namedRangeId", ""),
                    "range": range_str,
                }
            )

        return {"named_ranges": named_ranges_list, "count": len(named_ranges_list)}


class GoogleSheetsAddDropdownBlock(Block):
    """Add a dropdown (data validation) to cells."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        range: str = SchemaField(
            description="Cell range to add dropdown to (e.g., 'B2:B100')",
            placeholder="B2:B100",
        )
        options: list[str] = SchemaField(
            description="List of dropdown options",
        )
        strict: bool = SchemaField(
            description="Reject input not in the list",
            default=True,
        )
        show_dropdown: bool = SchemaField(
            description="Show dropdown arrow in cells",
            default=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the operation",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="725431c9-71ba-4fce-b829-5a3e495a8a88",
            description="Add a dropdown list (data validation) to cells. Useful for enforcing valid inputs.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAddDropdownBlock.Input,
            output_schema=GoogleSheetsAddDropdownBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "B2:B100",
                "options": ["Active", "Inactive", "Pending"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_add_dropdown": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        if not input_data.options:
            yield "error", "Options list cannot be empty"
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._add_dropdown,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.range,
                input_data.options,
                input_data.strict,
                input_data.show_dropdown,
            )
            yield "result", result
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to add dropdown: {str(e)}"

    def _add_dropdown(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        range_str: str,
        options: list[str],
        strict: bool,
        show_dropdown: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Parse range
        if "!" in range_str:
            range_str = range_str.split("!")[1]

        match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_str.upper())
        if not match:
            raise ValueError(f"Invalid range format: {range_str}")

        start_col = _column_letter_to_index(match.group(1))
        start_row = int(match.group(2)) - 1
        end_col = _column_letter_to_index(match.group(3)) + 1
        end_row = int(match.group(4))

        # Build condition values
        condition_values = [{"userEnteredValue": opt} for opt in options]

        request = {
            "setDataValidation": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": end_row,
                    "startColumnIndex": start_col,
                    "endColumnIndex": end_col,
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_LIST",
                        "values": condition_values,
                    },
                    "strict": strict,
                    "showCustomUi": show_dropdown,
                },
            }
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [request]}
        ).execute()

        return {"success": True}


class GoogleSheetsCopyToSpreadsheetBlock(Block):
    """Copy a sheet to another spreadsheet."""

    class Input(BlockSchemaInput):
        source_spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Source Spreadsheet",
            description="Select the source spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        source_sheet_name: str = SchemaField(
            description="Sheet to copy (optional, defaults to first sheet)",
            default="",
        )
        destination_spreadsheet_id: str = SchemaField(
            description="ID of the destination spreadsheet",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the copy operation",
        )
        new_sheet_id: int = SchemaField(
            description="ID of the new sheet in the destination",
        )
        new_sheet_name: str = SchemaField(
            description="Name of the new sheet",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The source spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="740eec3f-2b51-4e95-b87f-22ce2acafdfa",
            description="Copy a sheet from one spreadsheet to another.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsCopyToSpreadsheetBlock.Input,
            output_schema=GoogleSheetsCopyToSpreadsheetBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "source_spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Source Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "destination_spreadsheet_id": "dest_spreadsheet_id_123",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("new_sheet_id", 12345),
                ("new_sheet_name", "Copy of Sheet1"),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Source Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_copy_to_spreadsheet": lambda *args, **kwargs: {
                    "success": True,
                    "new_sheet_id": 12345,
                    "new_sheet_name": "Copy of Sheet1",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.source_spreadsheet:
            yield "error", "No source spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.source_spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._copy_to_spreadsheet,
                service,
                input_data.source_spreadsheet.id,
                input_data.source_sheet_name,
                input_data.destination_spreadsheet_id,
            )
            yield "result", {"success": True}
            yield "new_sheet_id", result["new_sheet_id"]
            yield "new_sheet_name", result["new_sheet_name"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.source_spreadsheet.id,
                name=input_data.source_spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.source_spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.source_spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to copy sheet: {str(e)}"

    def _copy_to_spreadsheet(
        self,
        service,
        source_spreadsheet_id: str,
        source_sheet_name: str,
        destination_spreadsheet_id: str,
    ) -> dict:
        target_sheet = resolve_sheet_name(
            service, source_spreadsheet_id, source_sheet_name or None
        )
        sheet_id = sheet_id_by_name(service, source_spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        result = (
            service.spreadsheets()
            .sheets()
            .copyTo(
                spreadsheetId=source_spreadsheet_id,
                sheetId=sheet_id,
                body={"destinationSpreadsheetId": destination_spreadsheet_id},
            )
            .execute()
        )

        return {
            "success": True,
            "new_sheet_id": result.get("sheetId", 0),
            "new_sheet_name": result.get("title", ""),
        }


class GoogleSheetsProtectRangeBlock(Block):
    """Protect a range from editing."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Sheet name (optional, defaults to first sheet)",
            default="",
        )
        range: str = SchemaField(
            description="Cell range to protect (e.g., 'A1:D10'). Leave empty to protect entire sheet.",
            default="",
        )
        description: str = SchemaField(
            description="Description for the protected range",
            default="Protected by automation",
        )
        warning_only: bool = SchemaField(
            description="Show warning but allow editing (vs blocking completely)",
            default=False,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(
            description="Result of the operation",
        )
        protection_id: int = SchemaField(
            description="ID of the protection",
        )
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining",
        )
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="d0e4f5d1-76e7-4082-9be8-e656ec1f432d",
            description="Protect a cell range or entire sheet from editing.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsProtectRangeBlock.Input,
            output_schema=GoogleSheetsProtectRangeBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "range": "A1:D10",
                "description": "Header row protection",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("protection_id", 12345),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_protect_range": lambda *args, **kwargs: {
                    "success": True,
                    "protection_id": 12345,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._protect_range,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.range,
                input_data.description,
                input_data.warning_only,
            )
            yield "result", {"success": True}
            yield "protection_id", result["protection_id"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to protect range: {str(e)}"

    def _protect_range(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        range_str: str,
        description: str,
        warning_only: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        protected_range: dict = {"sheetId": sheet_id}

        if range_str:
            # Parse specific range
            if "!" in range_str:
                range_str = range_str.split("!")[1]

            match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", range_str.upper())
            if not match:
                raise ValueError(f"Invalid range format: {range_str}")

            protected_range["startRowIndex"] = int(match.group(2)) - 1
            protected_range["endRowIndex"] = int(match.group(4))
            protected_range["startColumnIndex"] = _column_letter_to_index(
                match.group(1)
            )
            protected_range["endColumnIndex"] = (
                _column_letter_to_index(match.group(3)) + 1
            )

        request = {
            "addProtectedRange": {
                "protectedRange": {
                    "range": protected_range,
                    "description": description,
                    "warningOnly": warning_only,
                }
            }
        }

        result = (
            service.spreadsheets()
            .batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": [request]})
            .execute()
        )

        protection_id = 0
        replies = result.get("replies", [])
        if replies and "addProtectedRange" in replies[0]:
            protection_id = replies[0]["addProtectedRange"]["protectedRange"][
                "protectedRangeId"
            ]

        return {"success": True, "protection_id": protection_id}


class GoogleSheetsExportCsvBlock(Block):
    """Export a sheet as CSV data."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to export from",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            default="",
            description="Name of the sheet to export. Defaults to first sheet.",
        )
        include_headers: bool = SchemaField(
            default=True,
            description="Include the first row (headers) in the CSV output",
        )

    class Output(BlockSchemaOutput):
        csv_data: str = SchemaField(description="The sheet data as CSV string")
        row_count: int = SchemaField(description="Number of rows exported")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if export failed")

    def __init__(self):
        super().__init__(
            id="2617e68a-43b3-441f-8b11-66bb041105b8",
            description="Export a Google Sheet as CSV data",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsExportCsvBlock.Input,
            output_schema=GoogleSheetsExportCsvBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("csv_data", "Name,Email,Status\nJohn,john@test.com,Active\n"),
                ("row_count", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_export_csv": lambda *args, **kwargs: {
                    "csv_data": "Name,Email,Status\nJohn,john@test.com,Active\n",
                    "row_count": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._export_csv,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.include_headers,
            )
            yield "csv_data", result["csv_data"]
            yield "row_count", result["row_count"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to export CSV: {str(e)}"

    def _export_csv(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        include_headers: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        range_name = f"'{target_sheet}'"

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=range_name)
            .execute()
        )

        rows = result.get("values", [])

        # Skip header row if not including headers
        if not include_headers and rows:
            rows = rows[1:]

        output = io.StringIO()
        writer = csv.writer(output)
        for row in rows:
            writer.writerow(row)

        csv_data = output.getvalue()
        return {"csv_data": csv_data, "row_count": len(rows)}


class GoogleSheetsImportCsvBlock(Block):
    """Import CSV data into a sheet."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to import into",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        csv_data: str = SchemaField(description="CSV data to import")
        sheet_name: str = SchemaField(
            default="",
            description="Name of the sheet. Defaults to first sheet.",
        )
        start_cell: str = SchemaField(
            default="A1",
            description="Cell to start importing at (e.g., A1, B2)",
        )
        clear_existing: bool = SchemaField(
            default=False,
            description="Clear existing data before importing",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Import result")
        rows_imported: int = SchemaField(description="Number of rows imported")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if import failed")

    def __init__(self):
        super().__init__(
            id="cb992884-1ff2-450a-8f1b-7650d63e3aa0",
            description="Import CSV data into a Google Sheet",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsImportCsvBlock.Input,
            output_schema=GoogleSheetsImportCsvBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "csv_data": "Name,Email,Status\nJohn,john@test.com,Active\n",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                ("rows_imported", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_import_csv": lambda *args, **kwargs: {
                    "success": True,
                    "rows_imported": 2,
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._import_csv,
                service,
                input_data.spreadsheet.id,
                input_data.csv_data,
                input_data.sheet_name,
                input_data.start_cell,
                input_data.clear_existing,
            )
            yield "result", {"success": True}
            yield "rows_imported", result["rows_imported"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to import CSV: {str(e)}"

    def _import_csv(
        self,
        service,
        spreadsheet_id: str,
        csv_data: str,
        sheet_name: str,
        start_cell: str,
        clear_existing: bool,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)

        # Parse CSV data
        reader = csv.reader(io.StringIO(csv_data))
        rows = list(reader)

        if not rows:
            return {"success": True, "rows_imported": 0}

        # Clear existing data if requested
        if clear_existing:
            service.spreadsheets().values().clear(
                spreadsheetId=spreadsheet_id,
                range=f"'{target_sheet}'",
            ).execute()

        # Write data
        range_name = f"'{target_sheet}'!{start_cell}"
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            body={"values": rows},
        ).execute()

        return {"success": True, "rows_imported": len(rows)}


class GoogleSheetsAddNoteBlock(Block):
    """Add a note (comment) to a cell."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to add note to",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        cell: str = SchemaField(
            description="Cell to add note to (e.g., A1, B2)",
        )
        note: str = SchemaField(description="Note text to add")
        sheet_name: str = SchemaField(
            default="",
            description="Name of the sheet. Defaults to first sheet.",
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Result of the operation")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="774ac529-74f9-41da-bbba-6a06a51a5d7e",
            description="Add a note to a cell in a Google Sheet",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAddNoteBlock.Input,
            output_schema=GoogleSheetsAddNoteBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "cell": "A1",
                "note": "This is a test note",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_add_note": lambda *args, **kwargs: {"success": True},
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            await asyncio.to_thread(
                self._add_note,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.cell,
                input_data.note,
            )
            yield "result", {"success": True}
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to add note: {str(e)}"

    def _add_note(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        cell: str,
        note: str,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        sheet_id = sheet_id_by_name(service, spreadsheet_id, target_sheet)

        if sheet_id is None:
            raise ValueError(f"Sheet '{target_sheet}' not found")

        # Parse cell reference
        match = re.match(r"([A-Z]+)(\d+)", cell.upper())
        if not match:
            raise ValueError(f"Invalid cell reference: {cell}")

        col_index = _column_letter_to_index(match.group(1))
        row_index = int(match.group(2)) - 1

        request = {
            "updateCells": {
                "rows": [{"values": [{"note": note}]}],
                "fields": "note",
                "start": {
                    "sheetId": sheet_id,
                    "rowIndex": row_index,
                    "columnIndex": col_index,
                },
            }
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [request]}
        ).execute()

        return {"success": True}


class GoogleSheetsGetNotesBlock(Block):
    """Get notes from cells in a range."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to get notes from",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        range: str = SchemaField(
            default="A1:Z100",
            description="Range to get notes from (e.g., A1:B10)",
        )
        sheet_name: str = SchemaField(
            default="",
            description="Name of the sheet. Defaults to first sheet.",
        )

    class Output(BlockSchemaOutput):
        notes: list[dict] = SchemaField(description="List of notes with cell and text")
        count: int = SchemaField(description="Number of notes found")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="fa16834f-fff4-4d7a-9f7f-531ced90492b",
            description="Get notes from cells in a Google Sheet",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsGetNotesBlock.Input,
            output_schema=GoogleSheetsGetNotesBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "notes",
                    [
                        {"cell": "A1", "note": "Header note"},
                        {"cell": "B2", "note": "Data note"},
                    ],
                ),
                ("count", 2),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_get_notes": lambda *args, **kwargs: {
                    "notes": [
                        {"cell": "A1", "note": "Header note"},
                        {"cell": "B2", "note": "Data note"},
                    ],
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_sheets_service(credentials)
            result = await asyncio.to_thread(
                self._get_notes,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                input_data.range,
            )
            notes = result["notes"]
            yield "notes", notes
            yield "count", len(notes)
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to get notes: {str(e)}"

    def _get_notes(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        range_str: str,
    ) -> dict:

        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name or None)
        full_range = f"'{target_sheet}'!{range_str}"

        # Get spreadsheet data including notes
        result = (
            service.spreadsheets()
            .get(
                spreadsheetId=spreadsheet_id,
                ranges=[full_range],
                includeGridData=True,
            )
            .execute()
        )

        notes = []
        sheets = result.get("sheets", [])

        for sheet in sheets:
            data = sheet.get("data", [])
            for grid_data in data:
                start_row = grid_data.get("startRow", 0)
                start_col = grid_data.get("startColumn", 0)
                row_data = grid_data.get("rowData", [])

                for row_idx, row in enumerate(row_data):
                    values = row.get("values", [])
                    for col_idx, cell in enumerate(values):
                        note = cell.get("note")
                        if note:
                            col_letter = _index_to_column_letter(start_col + col_idx)
                            cell_ref = f"{col_letter}{start_row + row_idx + 1}"
                            notes.append({"cell": cell_ref, "note": note})

        return {"notes": notes}


class GoogleSheetsShareSpreadsheetBlock(Block):
    """Share a spreadsheet with specific users or make it accessible."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to share",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
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
        share_link: str = SchemaField(description="Link to the spreadsheet")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if share failed")

    def __init__(self):
        super().__init__(
            id="3e47e8ac-511a-4eb6-89c5-a6bcedc4236f",
            description="Share a Google Spreadsheet with users or get shareable link",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsShareSpreadsheetBlock.Input,
            output_schema=GoogleSheetsShareSpreadsheetBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "email": "test@example.com",
                "role": "reader",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True}),
                (
                    "share_link",
                    "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_share_spreadsheet": lambda *args, **kwargs: {
                    "success": True,
                    "share_link": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_drive_service(credentials)
            result = await asyncio.to_thread(
                self._share_spreadsheet,
                service,
                input_data.spreadsheet.id,
                input_data.email,
                input_data.role,
                input_data.send_notification,
                input_data.message,
            )
            yield "result", {"success": True}
            yield "share_link", result["share_link"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to share spreadsheet: {str(e)}"

    def _share_spreadsheet(
        self,
        service,
        spreadsheet_id: str,
        email: str,
        role: ShareRole,
        send_notification: bool,
        message: str,
    ) -> dict:
        share_link = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"

        if email:
            # Share with specific user
            permission = {"type": "user", "role": role.value, "emailAddress": email}

            kwargs: dict = {
                "fileId": spreadsheet_id,
                "body": permission,
                "sendNotificationEmail": send_notification,
            }
            if message:
                kwargs["emailMessage"] = message

            service.permissions().create(**kwargs).execute()
        else:
            # Get shareable link - use reader or commenter only (writer not allowed for "anyone")
            link_role = "reader" if role == ShareRole.WRITER else role.value
            permission = {"type": "anyone", "role": link_role}
            service.permissions().create(
                fileId=spreadsheet_id, body=permission
            ).execute()
            share_link += "?usp=sharing"

        return {"success": True, "share_link": share_link}


class GoogleSheetsSetPublicAccessBlock(Block):
    """Make a spreadsheet publicly accessible or private."""

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDriveFileField(
            title="Spreadsheet",
            description="The spreadsheet to modify access for",
            credentials_kwarg="credentials",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
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
        share_link: str = SchemaField(description="Link to the spreadsheet")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet for chaining"
        )
        error: str = SchemaField(description="Error message if operation failed")

    def __init__(self):
        super().__init__(
            id="d08d46cd-088b-4ba7-a545-45050f33b889",
            description="Make a Google Spreadsheet public or private",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsSetPublicAccessBlock.Input,
            output_schema=GoogleSheetsSetPublicAccessBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "public": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("result", {"success": True, "is_public": True}),
                (
                    "share_link",
                    "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?usp=sharing",
                ),
                (
                    "spreadsheet",
                    GoogleDriveFile(
                        id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                        name="Test Spreadsheet",
                        mimeType="application/vnd.google-apps.spreadsheet",
                        url="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit",
                        iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                        isFolder=False,
                        _credentials_id=None,
                    ),
                ),
            ],
            test_mock={
                "_set_public_access": lambda *args, **kwargs: {
                    "success": True,
                    "is_public": True,
                    "share_link": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?usp=sharing",
                },
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.spreadsheet:
            yield "error", "No spreadsheet selected"
            return

        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return

        try:
            service = _build_drive_service(credentials)
            result = await asyncio.to_thread(
                self._set_public_access,
                service,
                input_data.spreadsheet.id,
                input_data.public,
                input_data.role,
            )
            yield "result", {"success": True, "is_public": result["is_public"]}
            yield "share_link", result["share_link"]
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
                _credentials_id=input_data.spreadsheet.credentials_id,
            )
        except Exception as e:
            yield "error", f"Failed to set public access: {str(e)}"

    def _set_public_access(
        self,
        service,
        spreadsheet_id: str,
        public: bool,
        role: PublicAccessRole,
    ) -> dict:
        share_link = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"

        if public:
            # Make public
            permission = {"type": "anyone", "role": role.value}
            service.permissions().create(
                fileId=spreadsheet_id, body=permission
            ).execute()
            share_link += "?usp=sharing"
        else:
            # Make private - remove 'anyone' permissions
            permissions = service.permissions().list(fileId=spreadsheet_id).execute()
            for perm in permissions.get("permissions", []):
                if perm.get("type") == "anyone":
                    service.permissions().delete(
                        fileId=spreadsheet_id, permissionId=perm["id"]
                    ).execute()

        return {"success": True, "is_public": public, "share_link": share_link}
