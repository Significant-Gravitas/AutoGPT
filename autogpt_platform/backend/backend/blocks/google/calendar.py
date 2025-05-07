from datetime import datetime, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.settings import AppEnvironment, Settings

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)


class GoogleCalendarReadNextEventsBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/calendar.readonly"]
        )
        max_results: int = SchemaField(description="blah", default=10)
        calendar_id: str = SchemaField(
            description="The Calendar to Query", default="primary"
        )
        start_time: datetime = SchemaField(
            description="wehn to start, defaults to timezone utc now",
            default=datetime.now(tz=timezone.utc),
        )

    class Output(BlockSchema):
        events: list[dict] = SchemaField(description="", default_factory=list)
        error: str = SchemaField(
            description="Error message if any",
        )

    def __init__(self):
        settings = Settings()
        super().__init__(
            id="80bc3ed1-e9a4-449e-8163-a8fc86f74f6a",
            description="This block reads data from a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=GoogleCalendarReadNextEventsBlock.Input,
            output_schema=GoogleCalendarReadNextEventsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED
            or settings.config.app_env == AppEnvironment.PRODUCTION,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[],
            test_mock={
                "_read_calendar": lambda *args, **kwargs: [
                    # ["Name", "Score"],
                    # ["Alice", "85"],
                ],
            },
        )

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = self._build_service(credentials, **kwargs)
        data = self._read_calendar(
            service,
            calendarId=input_data.calendar_id,
            time_min=input_data.start_time.isoformat(),
            max_results=input_data.max_results,
            single_events=True,
        )
        yield "events", data

    @staticmethod
    def _build_service(credentials: GoogleCredentials, **kwargs):
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
            client_id=Settings().secrets.google_client_id,
            client_secret=Settings().secrets.google_client_secret,
            scopes=credentials.scopes,
        )
        return build("calendar", "v3", credentials=creds)

    def _read_calendar(
        self,
        service,
        calendarId: str,
        time_min: str,
        max_results: int,
        single_events: bool,
    ) -> list[list[str]]:
        calendar = service.events()
        result = calendar.list(
            calendarId=calendarId,
            timeMin=time_min,
            maxResults=max_results,
            singleEvents=single_events,
            orderBy="startTime",
        ).execute()
        return result.get("items", [])


# class GoogleSheetsWriteBlock(Block):
#     class Input(BlockSchema):
#         credentials: GoogleCredentialsInput = GoogleCredentialsField(
#             ["https://www.googleapis.com/auth/spreadsheets"]
#         )
#         spreadsheet_id: str = SchemaField(
#             description="The ID of the spreadsheet to write to",
#         )
#         range: str = SchemaField(
#             description="The A1 notation of the range to write",
#         )
#         values: list[list[str]] = SchemaField(
#             description="The data to write to the spreadsheet",
#         )

#     class Output(BlockSchema):
#         result: dict = SchemaField(
#             description="The result of the write operation",
#         )
#         error: str = SchemaField(
#             description="Error message if any",
#         )

#     def __init__(self):
#         super().__init__(
#             id="d9291e87-301d-47a8-91fe-907fb55460e5",
#             description="This block writes data to a Google Sheets spreadsheet.",
#             categories={BlockCategory.DATA},
#             input_schema=GoogleSheetsWriteBlock.Input,
#             output_schema=GoogleSheetsWriteBlock.Output,
#             disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
#             test_input={
#                 "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
#                 "range": "Sheet1!A1:B2",
#                 "values": [
#                     ["Name", "Score"],
#                     ["Bob", "90"],
#                 ],
#                 "credentials": TEST_CREDENTIALS_INPUT,
#             },
#             test_credentials=TEST_CREDENTIALS,
#             test_output=[
#                 (
#                     "result",
#                     {"updatedCells": 4, "updatedColumns": 2, "updatedRows": 2},
#                 ),
#             ],
#             test_mock={
#                 "_write_sheet": lambda *args, **kwargs: {
#                     "updatedCells": 4,
#                     "updatedColumns": 2,
#                     "updatedRows": 2,
#                 },
#             },
#         )

#     def run(
#         self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
#     ) -> BlockOutput:
#         service = GoogleSheetsReadBlock._build_service(credentials, **kwargs)
#         result = self._write_sheet(
#             service,
#             input_data.spreadsheet_id,
#             input_data.range,
#             input_data.values,
#         )
#         yield "result", result

#     def _write_sheet(
#         self, service, spreadsheet_id: str, range: str, values: list[list[str]]
#     ) -> dict:
#         body = {"values": values}
#         result = (
#             service.spreadsheets()
#             .values()
#             .update(
#                 spreadsheetId=spreadsheet_id,
#                 range=range,
#                 valueInputOption="USER_ENTERED",
#                 body=body,
#             )
#             .execute()
#         )
#         return result
