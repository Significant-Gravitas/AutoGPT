from autogpt_server.data.block import Block, BlockSchema, BlockOutput
from autogpt_server.data.model import SchemaField
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

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
