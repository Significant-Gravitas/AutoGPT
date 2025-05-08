import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel

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
        max_results: int = SchemaField(
            description="Max event count to load", default=10
        )
        calendar_id: str = SchemaField(
            description="The Calendar to Query", default="primary"
        )
        start_time: datetime = SchemaField(
            description="Time from which to start getting events, defaults to timezone utc now",
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
            description="This block reads events from a Google Calendar.",
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


class ReminderPreset(enum.Enum):
    """Common reminder times before an event."""

    TEN_MINUTES = 10
    THIRTY_MINUTES = 30
    ONE_HOUR = 60
    ONE_DAY = 1440  # 24 hours in minutes


class RecurrenceFrequency(enum.Enum):
    """Frequency options for recurring events."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


class ExactTiming(BaseModel):
    """Model for specifying start and end times."""

    discriminator: Literal["exact_timing"]
    start_datetime: datetime
    end_datetime: datetime


class DurationTiming(BaseModel):
    """Model for specifying start time and duration."""

    discriminator: Literal["duration_timing"]
    start_datetime: datetime
    duration_minutes: int


class OneTimeEvent(BaseModel):
    """Model for a one-time event."""

    discriminator: Literal["one_time"]


class RecurringEvent(BaseModel):
    """Model for a recurring event."""

    discriminator: Literal["recurring"]
    frequency: RecurrenceFrequency
    count: int


class GoogleCalendarCreateEventBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/calendar"]
        )
        # Event Details
        event_title: str = SchemaField(description="Title of the event")
        location: str | None = SchemaField(
            description="Location of the event", default=None
        )
        description: str | None = SchemaField(
            description="Description of the event", default=None
        )

        # Timing
        timing: ExactTiming | DurationTiming = SchemaField(
            discriminator="discriminator",
            advanced=False,
            description="Specify when the event starts and ends",
            default=DurationTiming(
                discriminator="duration_timing",
                start_datetime=datetime.now().replace(microsecond=0, second=0, minute=0)
                + timedelta(hours=1),
                duration_minutes=60,
            ),
        )

        # Calendar selection
        calendar_id: str = SchemaField(
            description="Calendar ID (use 'primary' for your main calendar)",
            default="primary",
        )

        # Guests
        guest_emails: list[str] = SchemaField(
            description="Email addresses of guests to invite", default_factory=list
        )
        send_notifications: bool = SchemaField(
            description="Send email notifications to guests", default=True
        )

        # Extras
        add_google_meet: bool = SchemaField(
            description="Include a Google Meet video conference link", default=False
        )
        recurrence: OneTimeEvent | RecurringEvent = SchemaField(
            discriminator="discriminator",
            description="Whether the event repeats",
            default=OneTimeEvent(discriminator="one_time"),
        )
        reminder_minutes: list[ReminderPreset] = SchemaField(
            description="When to send reminders before the event",
            default_factory=lambda: [ReminderPreset.TEN_MINUTES],
        )

    class Output(BlockSchema):
        event_id: str = SchemaField(description="ID of the created event")
        event_link: str = SchemaField(
            description="Link to view the event in Google Calendar"
        )
        error: str = SchemaField(description="Error message if event creation failed")

    def __init__(self):
        settings = Settings()
        # Generate a start time for testing (1 hour from now)
        test_start = datetime.now().replace(microsecond=0, second=0) + timedelta(
            hours=1
        )
        test_end = test_start + timedelta(hours=1)

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
                "event_title": "Team Meeting",
                "location": "Conference Room A",
                "description": "Weekly team sync-up",
                "timing": {
                    "discriminator": "exact_timing",
                    "start_datetime": test_start,
                    "end_datetime": test_end,
                },
                # "timezone": "America/Los_Angeles",
                "calendar_id": "primary",
                "guest_emails": ["colleague1@example.com", "colleague2@example.com"],
                "add_google_meet": True,
                "send_notifications": True,
                "recurrence": {
                    "discriminator": "recurring",
                    "frequency": RecurrenceFrequency.WEEKLY,
                    "count": 10,
                },
                "reminder_minutes": [
                    ReminderPreset.TEN_MINUTES,
                    ReminderPreset.ONE_HOUR,
                ],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("event_id", "abc123event_id"),
                ("event_link", "https://calendar.google.com/calendar/event?eid=abc123"),
            ],
            test_mock={
                "_create_event": lambda *args, **kwargs: {
                    "id": "abc123event_id",
                    "htmlLink": "https://calendar.google.com/calendar/event?eid=abc123",
                }
            },
        )

    def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            service = self._build_service(credentials, **kwargs)

            # Get start and end times based on the timing option
            if input_data.timing.discriminator == "exact_timing":
                start_datetime = input_data.timing.start_datetime
                end_datetime = input_data.timing.end_datetime
            else:  # duration_timing
                start_datetime = input_data.timing.start_datetime
                end_datetime = start_datetime + timedelta(
                    minutes=input_data.timing.duration_minutes
                )

            # Format datetimes for Google Calendar API
            start_time_str = start_datetime.isoformat()
            end_time_str = end_datetime.isoformat()

            # Build the event body
            event_body = {
                "summary": input_data.event_title,
                "start": {"dateTime": start_time_str},
                "end": {"dateTime": end_time_str},
            }

            # Add optional fields
            if input_data.location:
                event_body["location"] = input_data.location

            if input_data.description:
                event_body["description"] = input_data.description

            # Add guests
            if input_data.guest_emails:
                event_body["attendees"] = [
                    {"email": email} for email in input_data.guest_emails
                ]

            # Add reminders
            if input_data.reminder_minutes:
                event_body["reminders"] = {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": reminder.value}
                        for reminder in input_data.reminder_minutes
                    ],
                }

            # Add Google Meet
            if input_data.add_google_meet:
                event_body["conferenceData"] = {
                    "createRequest": {
                        "requestId": f"meet-{uuid.uuid4()}",
                        "conferenceSolutionKey": {"type": "hangoutsMeet"},
                    }
                }

            # Add recurrence
            if input_data.recurrence.discriminator == "recurring":
                rule = f"RRULE:FREQ={input_data.recurrence.frequency.value}"
                rule += f";COUNT={input_data.recurrence.count}"
                event_body["recurrence"] = [rule]

            # Create the event
            result = self._create_event(
                service=service,
                calendar_id=input_data.calendar_id,
                event_body=event_body,
                send_notifications=input_data.send_notifications,
                conference_data_version=1 if input_data.add_google_meet else 0,
            )

            yield "event_id", result.get("id", "")
            yield "event_link", result.get("htmlLink", "")
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
        event_body: dict,
        send_notifications: bool = False,
        conference_data_version: int = 0,
    ) -> dict:
        """Create a new event in Google Calendar."""
        calendar = service.events()

        # Make the API call
        result = calendar.insert(
            calendarId=calendar_id,
            body=event_body,
            sendNotifications=send_notifications,
            conferenceDataVersion=conference_data_version,
        ).execute()

        return result
