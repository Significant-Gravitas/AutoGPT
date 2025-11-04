from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.blocks.io import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    SchemaField,
)
from backend.data.model import GoogleDriveFile, GoogleDrivePickerField
from backend.sdk import BlockSchemaOutput


class GoogleSheetsReadTestBlock(Block):
    """
    Reads data from a Google Sheets spreadsheet.

    Uses the Google Drive Picker to select a spreadsheet, which provides
    the OAuth access token needed to read the sheet data.
    """

    class Input(BlockSchemaInput):
        spreadsheet: GoogleDriveFile = GoogleDrivePickerField(
            title="Spreadsheet",
            description="Select a Google Sheets spreadsheet",
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )

        range: str = SchemaField(
            default="Sheet1!A1:Z1000",
            description="Range to read (e.g., 'Sheet1!A1:Z1000')",
            placeholder="Sheet1!A1:Z1000",
        )

    class Output(BlockSchemaOutput):
        result: list[list[str]] = SchemaField(
            description="The data read from the spreadsheet",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="5724e902-3635-47e9-a108-aaa0263a4388",
            description="This block reads data from a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsReadTestBlock.Input,
            output_schema=GoogleSheetsReadTestBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Read data from a Google Sheets spreadsheet using the access token
        from the Drive Picker.
        """
        try:
            # Validate that we have the necessary data
            if not input_data.spreadsheet:
                yield "error", "No spreadsheet selected"
                return

            if not input_data.spreadsheet.access_token:
                yield "error", "No access token available. Please re-select the file."
                return

            spreadsheet_id = input_data.spreadsheet.id

            # Create credentials from the access token provided by the picker
            credentials = Credentials(token=input_data.spreadsheet.access_token)

            # Build the Sheets API service
            service = build(
                "sheets", "v4", credentials=credentials, cache_discovery=False
            )

            # Read the sheet data
            values = self._read_sheet(service, spreadsheet_id, input_data.range)

            yield "result", values

        except Exception as e:
            yield "error", f"Failed to read Google Sheet: {str(e)}"

    def _read_sheet(
        self, service, spreadsheet_id: str, range_str: str
    ) -> list[list[str]]:
        """Helper method to read sheet data"""
        sheet = service.spreadsheets()
        result = (
            sheet.values().get(spreadsheetId=spreadsheet_id, range=range_str).execute()
        )
        return result.get("values", [])
