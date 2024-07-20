from autogpt_server.data.block import Block, BlockSchema, BlockOutput
from autogpt_server.data.model import SchemaField
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

class GoogleSheetsWriter(Block):
    class Input(BlockSchema):
        spreadsheet_id: str = SchemaField(
            description="The ID of the Google Sheet to write to",
            placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        )
        sheet_name: str = SchemaField(
            description="The name of the sheet to append data to",
            placeholder="Sheet1",
            default="Sheet1"
        )
        row_data: list = SchemaField(
            description="The data to append as a single row",
            placeholder="['John Doe', 'johndoe@example.com', '30']"
        )
        access_token: str = SchemaField(
            description="Google OAuth2 access token. You can obtain this from the Google Cloud Console.",
            placeholder="ya29.a0AbVbY6P...",
            secret=True
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
            credentials = Credentials(input_data.access_token)
            service = build("sheets", "v4", credentials=credentials)
            sheet = service.spreadsheets()

            # Get the current sheet data to find the next empty row
            result = sheet.values().get(
                spreadsheetId=input_data.spreadsheet_id,
                range=f"{input_data.sheet_name}!A:A"
            ).execute()
            values = result.get('values', [])
            next_row = len(values) + 1

            # Prepare the range for appending
            append_range = f"{input_data.sheet_name}!A{next_row}"

            body = {
                'values': [input_data.row_data]
            }

            result = sheet.values().append(
                spreadsheetId=input_data.spreadsheet_id,
                range=append_range,
                valueInputOption='USER_ENTERED',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            yield "result", {
                "success": True,
                "updated_range": result.get('updates', {}).get('updatedRange'),
                "updated_rows": result.get('updates', {}).get('updatedRows'),
                "updated_cells": result.get('updates', {}).get('updatedCells')
            }

        except Exception as e:
            yield "result", {
                "success": False,
                "error": str(e)
            }

class GoogleSheetsReader(Block):
    class Input(BlockSchema):
        spreadsheet_id: str = SchemaField(
            description="The ID of the Google Sheet to read",
            placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        )
        range: str = SchemaField(
            description="The A1 notation of the range to read",
            placeholder="Sheet1!A1:E10",
            default= "Sheet1!A1:Z1000"
        )
        access_token: str = SchemaField(
            description="Google OAuth2 access token",
            secret=True
        )

    class Output(BlockSchema):
        rows: list = SchemaField(description="The rows of from the specified Google Sheet range")

    def __init__(self):
        super().__init__(
            id="google-sheets-block",
            input_schema=GoogleSheetsReader.Input,
            output_schema=GoogleSheetsReader.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        credentials = Credentials(input_data.access_token)
        service = build("sheets", "v4", credentials=credentials)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=input_data.spreadsheet_id, range=input_data.range).execute()
        values = result.get("values", [])
        
        if not values:
            yield "data", {"error": "No data found."}
        else:
            data = [row for row in values[0:]]
            yield "rows", data
