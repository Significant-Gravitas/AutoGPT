import asyncio
from enum import Enum
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.blocks.google._drive import GoogleDriveFile, GoogleDrivePickerField
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


def _convert_dicts_to_rows(
    data: list[dict[str, Any]], headers: list[str]
) -> list[list[str]]:
    """Convert list of dictionaries to list of rows using the specified header order.

    Args:
        data: List of dictionaries to convert
        headers: List of column headers to use for ordering

    Returns:
        List of rows where each row is a list of string values in header order
    """
    if not data:
        return []

    if not headers:
        raise ValueError("Headers are required when using list[dict] format")

    rows = []
    for item in data:
        row = []
        for header in headers:
            value = item.get(header, "")
            row.append(str(value) if value is not None else "")
        rows.append(row)

    return rows


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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=spreadsheet_id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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


class GoogleSheetsAppendBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )
        sheet_name: str = SchemaField(
            description="Optional sheet to append to (defaults to first sheet)",
            default="",
        )
        values: list[list[str]] = SchemaField(
            description="Rows to append as list of rows (list[list[str]])",
            default=[],
        )
        dict_values: list[dict[str, Any]] = SchemaField(
            description="Rows to append as list of dictionaries (list[dict])",
            default=[],
        )
        headers: list[str] = SchemaField(
            description="Column headers to use for ordering dict values (required when dict_values is provided)",
            default=[],
        )
        range: str = SchemaField(
            description="Range to append to (e.g. 'A:A' for column A only, 'A:C' for columns A-C, or leave empty for unlimited columns). When empty, data will span as many columns as needed.",
            default="",
            advanced=True,
        )
        value_input_option: ValueInputOption = SchemaField(
            description="How input data should be interpreted",
            default=ValueInputOption.USER_ENTERED,
            advanced=True,
        )
        insert_data_option: InsertDataOption = SchemaField(
            description="How new data should be inserted",
            default=InsertDataOption.INSERT_ROWS,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        result: dict = SchemaField(description="Append API response")
        spreadsheet: GoogleDriveFile = SchemaField(
            description="The spreadsheet as a GoogleDriveFile (for chaining to other blocks)",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="531d50c0-d6b9-4cf9-a013-7bf783d313c7",
            description="Append data to a Google Sheet. Use 'values' for list of rows (list[list[str]]) or 'dict_values' with 'headers' for list of dictionaries (list[dict]). Data is added to the next empty row without overwriting existing content. Leave range empty for unlimited columns, or specify range like 'A:A' to constrain to specific columns.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsAppendBlock.Input,
            output_schema=GoogleSheetsAppendBlock.Output,
            disabled=GOOGLE_SHEETS_DISABLED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "spreadsheet": {
                    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                    "name": "Test Spreadsheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                },
                "values": [["Charlie", "95"]],
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
                    ),
                ),
            ],
            test_mock={
                "_append_sheet": lambda *args, **kwargs: {
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

        # Check if the selected file is actually a Google Sheets spreadsheet
        validation_error = _validate_spreadsheet_file(input_data.spreadsheet)
        if validation_error:
            yield "error", validation_error
            return
        try:
            service = _build_sheets_service(credentials)

            # Determine which values to use and convert if needed
            processed_values: list[list[str]]

            # Validate that only one format is provided
            if input_data.values and input_data.dict_values:
                raise ValueError("Provide either 'values' or 'dict_values', not both")

            if input_data.dict_values:
                if not input_data.headers:
                    raise ValueError("Headers are required when using dict_values")
                processed_values = _convert_dicts_to_rows(
                    input_data.dict_values, input_data.headers
                )
            elif input_data.values:
                processed_values = input_data.values
            else:
                raise ValueError("Either 'values' or 'dict_values' must be provided")

            result = await asyncio.to_thread(
                self._append_sheet,
                service,
                input_data.spreadsheet.id,
                input_data.sheet_name,
                processed_values,
                input_data.range,
                input_data.value_input_option,
                input_data.insert_data_option,
            )
            yield "result", result
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
            )
        except Exception as e:
            yield "error", f"Failed to append to Google Sheet: {str(e)}"

    def _append_sheet(
        self,
        service,
        spreadsheet_id: str,
        sheet_name: str,
        values: list[list[str]],
        range: str,
        value_input_option: ValueInputOption,
        insert_data_option: InsertDataOption,
    ) -> dict:
        target_sheet = resolve_sheet_name(service, spreadsheet_id, sheet_name)
        formatted_sheet = format_sheet_name(target_sheet)
        # If no range specified, use A1 to let Google Sheets find the next empty row with unlimited columns
        # If range specified, use it to constrain columns (e.g., A:A for column A only)
        if range:
            append_range = f"{formatted_sheet}!{range}"
        else:
            # Use A1 as starting point for unlimited columns - Google Sheets will find next empty row
            append_range = f"{formatted_sheet}!A1"
        body = {"values": values}
        return (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range=append_range,
                valueInputOption=value_input_option.value,
                insertDataOption=insert_data_option.value,
                body=body,
            )
            .execute()
        )


class GoogleSheetsClearBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            # Output the GoogleDriveFile for chaining
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
                yield "spreadsheet", GoogleDriveFile(
                    id=input_data.spreadsheet.id,
                    name=input_data.spreadsheet.name,
                    mimeType="application/vnd.google-apps.spreadsheet",
                    url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                    iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                    isFolder=False,
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
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
                "title": "Test Spreadsheet",
                "sheet_names": ["Sheet1", "Data", "Summary"],
                "credentials": TEST_CREDENTIALS_INPUT,
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
        service = _build_sheets_service(credentials)
        result = await asyncio.to_thread(
            self._create_spreadsheet,
            service,
            input_data.title,
            input_data.sheet_names,
        )

        if "error" in result:
            yield "error", result["error"]
        else:
            spreadsheet_id = result["spreadsheetId"]
            spreadsheet_url = result["spreadsheetUrl"]
            # Output the full GoogleDriveFile object for easy chaining
            yield "spreadsheet", GoogleDriveFile(
                id=spreadsheet_id,
                name=result.get("title", input_data.title),
                mimeType="application/vnd.google-apps.spreadsheet",
                url=spreadsheet_url,
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
            )
            yield "spreadsheet_id", spreadsheet_id
            yield "spreadsheet_url", spreadsheet_url
            yield "result", {"success": True}

    def _create_spreadsheet(self, service, title: str, sheet_names: list[str]) -> dict:
        try:
            # Create the initial spreadsheet
            spreadsheet_body = {
                "properties": {"title": title},
                "sheets": [
                    {
                        "properties": {
                            "title": sheet_names[0] if sheet_names else "Sheet1"
                        }
                    }
                ],
            }

            result = service.spreadsheets().create(body=spreadsheet_body).execute()
            spreadsheet_id = result["spreadsheetId"]
            spreadsheet_url = result["spreadsheetUrl"]

            # Add additional sheets if requested
            if len(sheet_names) > 1:
                requests = []
                for sheet_name in sheet_names[1:]:
                    requests.append({"addSheet": {"properties": {"title": sheet_name}}})

                if requests:
                    batch_body = {"requests": requests}
                    service.spreadsheets().batchUpdate(
                        spreadsheetId=spreadsheet_id, body=batch_body
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
        credentials: GoogleCredentialsInput = GoogleCredentialsField([])
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
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
                "credentials": TEST_CREDENTIALS_INPUT,
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
            yield "spreadsheet", GoogleDriveFile(
                id=input_data.spreadsheet.id,
                name=input_data.spreadsheet.name,
                mimeType="application/vnd.google-apps.spreadsheet",
                url=f"https://docs.google.com/spreadsheets/d/{input_data.spreadsheet.id}/edit",
                iconUrl="https://www.gstatic.com/images/branding/product/1x/sheets_48dp.png",
                isFolder=False,
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
