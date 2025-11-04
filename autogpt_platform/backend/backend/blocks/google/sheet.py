from backend.blocks.google._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from backend.blocks.io import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    SchemaField,
)
from backend.data.model import GoogleDrivePicker, GoogleDrivePickerField
from backend.sdk import BlockSchemaOutput


class GoogleSheetsReadTestBlock(Block):
    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )
        drive_id: GoogleDrivePicker = GoogleDrivePickerField(
            allow_folder_selection=True,
            description="The Google Sheets file to read from",
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
            test_input={
                "drive_id": "1234567890",
                "range": "Sheet1!A1:B2",
                "credentials": TEST_CREDENTIALS_INPUT,
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

        yield "result", input_data.drive_id

    def _read_sheet(self, service, spreadsheet_id: str, range: str) -> list[list[str]]:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range).execute()
        return result.get("values", [])
