import re
from typing import List, Union

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


def extract_spreadsheet_id(url: str) -> str:
    """Extract the spreadsheet ID from a Google Sheets URL."""
    match = re.search(
        r"/d/([a-zA-Z0-9-_]+)", url
    )  # This works for clean urls and those with extra parameters at the end
    if match:
        return match.group(1)
    raise ValueError("Invalid Google Sheets URL")


class GoogleSheetsWriter(Block):
    class Input(BlockSchema):
        spreadsheet_url: str = SchemaField(
            description="The link to the Google Sheet to write to.",
            placeholder="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        )
        sheet_name: str = SchemaField(
            description="The name of the sheet to append data to",
            placeholder="Sheet1",
            default="Sheet1",
        )
        row_data: list = SchemaField(
            description="The data to append as a single row",
            placeholder="['John Doe', 'johndoe@example.com', '30']",
        )
        access_token: str = SchemaField(
            description="Google OAuth2 access token. Click 'Sign in with Google' below to automatically generate this token. Keep this secret!",
            placeholder="Sign in with Google below",
            secret=True,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(description="The result of the append operation")

    def __init__(self):
        super().__init__(
            id="google-sheets-writer",
            input_schema=GoogleSheetsWriter.Input,
            output_schema=GoogleSheetsWriter.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            spreadsheet_id = extract_spreadsheet_id(input_data.spreadsheet_url)
            credentials = Credentials(input_data.access_token)
            service = build("sheets", "v4", credentials=credentials)
            sheet = service.spreadsheets()

            # Get the current sheet data to find the next empty row
            result = (
                sheet.values()
                .get(spreadsheetId=spreadsheet_id, range=f"{input_data.sheet_name}!A:A")
                .execute()
            )
            values = result.get("values", [])
            next_row = len(values) + 1

            # Prepare the range for appending
            append_range = f"{input_data.sheet_name}!A{next_row}"

            body = {"values": [input_data.row_data]}

            result = (
                sheet.values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=append_range,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )

            yield "result", {
                "success": True,
                "updated_range": result.get("updates", {}).get("updatedRange"),
                "updated_rows": result.get("updates", {}).get("updatedRows"),
                "updated_cells": result.get("updates", {}).get("updatedCells"),
            }

        except Exception as e:
            yield "result", {"success": False, "error": str(e)}


class GoogleSheetsReader(Block):
    class Input(BlockSchema):
        spreadsheet_url: str = SchemaField(
            description="The link to the Google Sheet to read from.",
            placeholder="https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        )
        read_single_row: bool = SchemaField(
            description="Do you want to read just one row? Select 'True' for a single row, 'False' for a range.",
            default=False,
        )
        sheet_name: str = SchemaField(
            description="The name of the sheet you want to read from (e.g., 'Sheet1')",
            placeholder="Sheet1",
            default="Sheet1",
        )
        row_number: int = SchemaField(
            description="If reading a single row, which row number do you want? (e.g., 3 for the third row)",
            placeholder="3",
        )
        cell_range: str = SchemaField(
            description="If reading a range, what range of cells do you want? (e.g., 'A1:E10' for the first 10 rows of columns A to E)",
            placeholder="A1:E10",
            default="A1:Z1000",
        )
        access_token: str = SchemaField(
            placeholder="Sign in with Google below",
            description="Google OAuth2 access token. Click 'Sign in with Google' below to automatically generate this token. Keep this secret!",
            secret=True,
        )

    class Output(BlockSchema):
        data: Union[List[str], List[List[str]]] = SchemaField(
            description="The information read from your Google Sheet. For a single row, it's a simple list. For multiple rows, it's a list of lists."
        )

    def __init__(self):
        super().__init__(
            id="google-sheets-reader",
            input_schema=GoogleSheetsReader.Input,
            output_schema=GoogleSheetsReader.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            spreadsheet_id = extract_spreadsheet_id(input_data.spreadsheet_url)
            credentials = Credentials(input_data.access_token)
            service = build("sheets", "v4", credentials=credentials)
            sheet = service.spreadsheets()

            if input_data.read_single_row:
                range_to_read = f"{input_data.sheet_name}!A{input_data.row_number}:ZZ{input_data.row_number}"
            else:
                range_to_read = f"{input_data.sheet_name}!{input_data.cell_range}"

            result = (
                sheet.values()
                .get(spreadsheetId=spreadsheet_id, range=range_to_read)
                .execute()
            )
            values = result.get("values", [])

            if not values:
                yield "data", {
                    "message": "No data found in the specified range or row."
                }
            else:
                if input_data.read_single_row:
                    # Return a single list for a single row
                    yield "data", values[0] if values else []
                else:
                    # Return a list of lists for multiple rows
                    yield "data", values

        except Exception as e:
            yield "data", {"error": f"Oops! Something went wrong: {str(e)}"}
