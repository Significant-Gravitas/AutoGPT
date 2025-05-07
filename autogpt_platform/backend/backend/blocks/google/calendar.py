from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, Optional
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


class EventDateTime(BaseModel):
    """Model for event date and time."""

    dateTime: str = SchemaField(
        description="Date and time in ISO format with timezone (e.g. '2025-05-28T09:00:00-07:00')"
    )
    timeZone: str | None = SchemaField(
        description="Time zone (e.g. 'America/Los_Angeles')", default=None
    )


class EventAttendee(BaseModel):
    """Model for event attendee."""

    email: str = SchemaField(description="Email address of attendee")
    optional: bool | None = SchemaField(
        description="Whether attendance is optional", default=None
    )
    responseStatus: str | None = SchemaField(
        description="Attendee's response status (needsAction, declined, tentative, accepted)",
        default=None,
    )


class EventReminder(BaseModel):
    """Model for event reminder."""

    method: str = SchemaField(description="Reminder method (email, popup)")
    minutes: int = SchemaField(description="Minutes before event to trigger reminder")


class EventReminders(BaseModel):
    """Model for event reminders configuration."""

    useDefault: bool = SchemaField(description="Whether to use default reminders")
    overrides: list[EventReminder] = SchemaField(
        description="list of custom reminders", default_factory=list
    )


class EventConferenceData(BaseModel):
    """Model for conference data."""

    createRequest: Optional[dict] = SchemaField(
        description="Request to create a conference", default=None
    )


class EventData(BaseModel):
    """Model for Google Calendar event data."""

    summary: str = SchemaField(description="Title of the event")
    location: str | None = SchemaField(
        description="Location of the event", default=None
    )
    description: str | None = SchemaField(
        description="Description of the event", default=None
    )
    start: EventDateTime
    end: EventDateTime
    recurrence: list[str] = SchemaField(
        description="Recurrence rules (e.g. ['RRULE:FREQ=DAILY;COUNT=2'])",
        default_factory=list,
    )
    attendees: list[EventAttendee] = SchemaField(
        description="list of attendees", default_factory=list
    )
    reminders: EventReminders = SchemaField(description="Reminders configuration")
    conferenceData: EventConferenceData = SchemaField(
        description="Conference data for video meetings"
    )


class GoogleCalendarCreateEventBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/calendar"]
        )
        calendar_id: str = SchemaField(
            description="Calendar ID to create the event in (use 'primary' for the user's primary calendar)",
            default="primary",
        )
        event: EventData = SchemaField(
            description="Event data for creating the calendar event"
        )
        send_notifications: bool = SchemaField(
            description="Whether to send notifications to attendees", default=False
        )
        supports_attachments: bool = SchemaField(
            description="Whether the request supports attachments", default=False
        )
        conference_data_version: int = SchemaField(
            description="Version for conference data support (0=no conference, 1=create conference)",
            default=0,
        )

    class Output(BlockSchema):
        class EventOutput(BaseModel):
            id: str = SchemaField(description="")
            htmlLink: str = SchemaField(description="")
            summary: str = SchemaField(description="")
            created: str = SchemaField(description="")
            updated: str = SchemaField(description="")

        event: EventOutput = SchemaField(
            description="Created event data including ID and link"
        )
        error: str = SchemaField(description="Error message if event creation failed")

    def __init__(self):
        settings = Settings()
        super().__init__(
            id="ed2ec950-fbff-4204-94c0-023fb1d625e0",
            description="This block creates a new event in Google Calendar with customizable parameters.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarCreateEventBlock.Input,
            output_schema=GoogleCalendarCreateEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED
            or settings.config.app_env == AppEnvironment.PRODUCTION,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "calendar_id": "primary",
                "event": {
                    "summary": "Test Event",
                    "location": "123 Test St, Test City",
                    "description": "This is a test event created by AutoGPT",
                    "start": {
                        "dateTime": "2025-05-28T09:00:00-07:00",
                        "timeZone": "America/Los_Angeles",
                    },
                    "end": {
                        "dateTime": "2025-05-28T10:00:00-07:00",
                        "timeZone": "America/Los_Angeles",
                    },
                    "attendees": [
                        {"email": "attendee1@example.com"},
                        {"email": "attendee2@example.com"},
                    ],
                    "reminders": {
                        "useDefault": False,
                        "overrides": [
                            {"method": "email", "minutes": 24 * 60},
                            {"method": "popup", "minutes": 10},
                        ],
                    },
                },
                "send_notifications": True,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "event",
                    {
                        "id": "abc123event_id",
                        "htmlLink": "https://calendar.google.com/calendar/event?eid=abc123",
                        "summary": "Test Event",
                        "created": "2025-05-01T12:00:00Z",
                        "updated": "2025-05-01T12:00:00Z",
                    },
                )
            ],
            test_mock={
                "_create_event": lambda *args, **kwargs: {
                    "id": "abc123event_id",
                    "htmlLink": "https://calendar.google.com/calendar/event?eid=abc123",
                    "summary": "Test Event",
                    "created": "2025-05-01T12:00:00Z",
                    "updated": "2025-05-01T12:00:00Z",
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            service = self._build_service(credentials, **kwargs)

            # # Convert Pydantic model to dict for API call
            event_data = input_data.event.dict(exclude_none=True)

            result = self._create_event(
                service=service,
                calendar_id=input_data.calendar_id,
                event_data=event_data,
                send_notifications=input_data.send_notifications,
                supports_attachments=input_data.supports_attachments,
                conference_data_version=input_data.conference_data_version,
            )

            # Extract relevant fields for output
            output_event = {
                "id": result.get("id", ""),
                "htmlLink": result.get("htmlLink", ""),
                "summary": result.get("summary", ""),
                "created": result.get("created", ""),
                "updated": result.get("updated", ""),
            }

            yield "event", output_event

        except Exception as e:
            yield "error", str(e)

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

    def _create_event(
        self,
        service,
        calendar_id: str,
        event_data: dict,
        send_notifications: bool = False,
        supports_attachments: bool = False,
        conference_data_version: int = 0,
    ) -> dict:
        """Create a new event in Google Calendar."""
        calendar = service.events()

        # Make the API call
        result = calendar.insert(
            calendarId=calendar_id,
            body=event_data,
            sendNotifications=send_notifications,
            supportsAttachments=supports_attachments,
            conferenceDataVersion=conference_data_version,
        ).execute()

        return result


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
