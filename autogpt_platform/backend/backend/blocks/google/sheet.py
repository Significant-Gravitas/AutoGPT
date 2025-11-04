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


class GoogleSheetsMultiReadTestBlock(Block):
    """
    Reads data from multiple Google Sheets spreadsheets.

    Uses the Google Drive Picker with multiselect enabled to select
    multiple spreadsheets at once.
    """

    class Input(BlockSchemaInput):
        spreadsheets: list[GoogleDriveFile] = GoogleDrivePickerField(
            title="Spreadsheets",
            description="Select multiple Google Sheets spreadsheets",
            multiselect=True,
            allowed_views=["SPREADSHEETS"],
            allowed_mime_types=["application/vnd.google-apps.spreadsheet"],
        )

        range: str = SchemaField(
            default="A1:Z1000",
            description="Range to read from each spreadsheet (e.g., 'A1:Z1000'). Will read from the first sheet if no sheet name is specified.",
            placeholder="A1:Z1000",
        )

        sheet_name: str = SchemaField(
            default="",
            description="Optional: specific sheet name to read from. If empty, reads from first sheet in each spreadsheet.",
            placeholder="Sheet1",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[dict] = SchemaField(
            description="List of results from each spreadsheet with name and data",
        )
        combined_data: list[list[str]] = SchemaField(
            description="All data combined from all spreadsheets",
        )
        total_spreadsheets: int = SchemaField(
            description="Total number of spreadsheets processed",
        )
        total_rows: int = SchemaField(
            description="Total number of rows read across all spreadsheets",
        )
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-4789-a012-345678901234",
            description="Read data from multiple Google Sheets spreadsheets at once.",
            categories={BlockCategory.DATA},
            input_schema=GoogleSheetsMultiReadTestBlock.Input,
            output_schema=GoogleSheetsMultiReadTestBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Read data from multiple Google Sheets spreadsheets.
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            num_sheets = len(input_data.spreadsheets) if input_data.spreadsheets else 0
            logger.info(
                f"GoogleSheetsMultiReadTestBlock: Processing {num_sheets} spreadsheets"
            )

            if not input_data.spreadsheets or num_sheets == 0:
                yield "error", "No spreadsheets selected"
                return

            results = []
            combined_data = []

            for idx, spreadsheet in enumerate(input_data.spreadsheets):
                logger.info(
                    f"Processing {idx + 1}/{num_sheets}: {spreadsheet.name or spreadsheet.id}"
                )

                if not spreadsheet.access_token:
                    logger.error(f"No access token for spreadsheet {idx + 1}")
                    yield "error", f"No access token for {spreadsheet.name or spreadsheet.id}"
                    return

                # Create credentials for this spreadsheet
                credentials = Credentials(token=spreadsheet.access_token)
                service = build(
                    "sheets", "v4", credentials=credentials, cache_discovery=False
                )

                # Read the sheet data
                try:
                    # Get the first sheet name if not specified
                    sheet_range = input_data.range
                    if input_data.sheet_name:
                        # User specified a sheet name
                        if "!" not in sheet_range:
                            sheet_range = f"{input_data.sheet_name}!{sheet_range}"
                    else:
                        # Get the first sheet name dynamically
                        if "!" not in sheet_range:
                            first_sheet_name = self._get_first_sheet_name(
                                service, spreadsheet.id
                            )
                            sheet_range = f"{first_sheet_name}!{sheet_range}"

                    values = self._read_sheet(service, spreadsheet.id, sheet_range)

                    logger.info(f"Read {len(values)} rows from {spreadsheet.name}")

                    results.append(
                        {
                            "name": spreadsheet.name or spreadsheet.id,
                            "id": spreadsheet.id,
                            "data": values,
                            "row_count": len(values),
                        }
                    )

                    # Add to combined data
                    combined_data.extend(values)

                except Exception as e:
                    logger.error(f"Error reading {spreadsheet.name}: {str(e)}")
                    results.append(
                        {
                            "name": spreadsheet.name or spreadsheet.id,
                            "id": spreadsheet.id,
                            "error": str(e),
                        }
                    )

            logger.info(
                f"Completed: {len(results)} spreadsheets, {len(combined_data)} total rows"
            )

            yield "results", results
            yield "combined_data", combined_data
            yield "total_spreadsheets", len(results)
            yield "total_rows", len(combined_data)

        except Exception as e:
            yield "error", f"Failed to read Google Sheets: {str(e)}"

    def _get_first_sheet_name(self, service, spreadsheet_id: str) -> str:
        """Get the name of the first sheet in the spreadsheet"""
        meta = (
            service.spreadsheets()
            .get(spreadsheetId=spreadsheet_id, includeGridData=False)
            .execute()
        )
        first_sheet = meta["sheets"][0]["properties"]
        return first_sheet["title"]

    def _read_sheet(
        self, service, spreadsheet_id: str, range_str: str
    ) -> list[list[str]]:
        """Helper method to read sheet data"""
        sheet = service.spreadsheets()
        result = (
            sheet.values().get(spreadsheetId=spreadsheet_id, range=range_str).execute()
        )
        return result.get("values", [])
