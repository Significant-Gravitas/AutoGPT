import asyncio
import enum
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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


class CalendarEvent(BaseModel):
    """Structured representation of a Google Calendar event."""

    id: str
    title: str
    start_time: str
    end_time: str
    is_all_day: bool
    location: str | None
    description: str | None
    organizer: str | None
    attendees: list[str]
    has_video_call: bool
    video_link: str | None
    calendar_link: str
    is_recurring: bool


class GoogleCalendarReadEventsBlock(Block):
    class Input(BlockSchema):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            ["https://www.googleapis.com/auth/calendar.readonly"]
        )
        calendar_id: str = SchemaField(
            description="Calendar ID (use 'primary' for your main calendar)",
            default="primary",
        )
        max_events: int = SchemaField(
            description="Maximum number of events to retrieve", default=10
        )
        start_time: datetime = SchemaField(
            description="Retrieve events starting from this time",
            default_factory=lambda: datetime.now(tz=timezone.utc),
        )
        time_range_days: int = SchemaField(
            description="Number of days to look ahead for events", default=30
        )
        search_term: str | None = SchemaField(
            description="Optional search term to filter events by", default=None
        )

        page_token: str | None = SchemaField(
            description="Page token from previous request to get the next batch of events. You can use this if you have lots of events you want to process in a loop",
            default=None,
        )
        include_declined_events: bool = SchemaField(
            description="Include events you've declined", default=False
        )

    class Output(BlockSchema):
        events: list[CalendarEvent] = SchemaField(
            description="List of calendar events in the requested time range",
            default_factory=list,
        )
        event: CalendarEvent = SchemaField(
            description="One of the calendar events in the requested time range"
        )
        next_page_token: str | None = SchemaField(
            description="Token for retrieving the next page of events if more exist",
            default=None,
        )
        error: str = SchemaField(
            description="Error message if the request failed",
        )

    def __init__(self):
        # Create realistic test data for events
        test_now = datetime.now(tz=timezone.utc)
        test_tomorrow = test_now + timedelta(days=1)

        test_event_dict = {
            "id": "event1id",
            "title": "Team Meeting",
            "start_time": test_tomorrow.strftime("%Y-%m-%d %H:%M"),
            "end_time": (test_tomorrow + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
            "is_all_day": False,
            "location": "Conference Room A",
            "description": "Weekly team sync",
            "organizer": "manager@example.com",
            "attendees": ["colleague1@example.com", "colleague2@example.com"],
            "has_video_call": True,
            "video_link": "https://meet.google.com/abc-defg-hij",
            "calendar_link": "https://calendar.google.com/calendar/event?eid=event1id",
            "is_recurring": True,
        }

        super().__init__(
            id="80bc3ed1-e9a4-449e-8163-a8fc86f74f6a",
            description="Retrieves upcoming events from a Google Calendar with filtering options",
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleCalendarReadEventsBlock.Input,
            output_schema=GoogleCalendarReadEventsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "calendar_id": "primary",
                "max_events": 5,
                "start_time": test_now.isoformat(),
                "time_range_days": 7,
                "search_term": None,
                "include_declined_events": False,
                "page_token": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("event", test_event_dict),
                ("events", [test_event_dict]),
            ],
            test_mock={
                "_read_calendar": lambda *args, **kwargs: {
                    "items": [
                        {
                            "id": "event1id",
                            "summary": "Team Meeting",
                            "start": {
                                "dateTime": test_tomorrow.isoformat(),
                                "timeZone": "UTC",
                            },
                            "end": {
                                "dateTime": (
                                    test_tomorrow + timedelta(hours=1)
                                ).isoformat(),
                                "timeZone": "UTC",
                            },
                            "location": "Conference Room A",
                            "description": "Weekly team sync",
                            "organizer": {"email": "manager@example.com"},
                            "attendees": [
                                {"email": "colleague1@example.com"},
                                {"email": "colleague2@example.com"},
                            ],
                            "conferenceData": {
                                "conferenceUrl": "https://meet.google.com/abc-defg-hij"
                            },
                            "htmlLink": "https://calendar.google.com/calendar/event?eid=event1id",
                            "recurrence": ["RRULE:FREQ=WEEKLY;COUNT=10"],
                        }
                    ],
                    "nextPageToken": None,
                },
                "_format_events": lambda *args, **kwargs: [test_event_dict],
            },
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            service = self._build_service(credentials, **kwargs)

            # Calculate end time based on start time and time range
            end_time = input_data.start_time + timedelta(
                days=input_data.time_range_days
            )

            # Call Google Calendar API
            result = await asyncio.to_thread(
                self._read_calendar,
                service=service,
                calendarId=input_data.calendar_id,
                time_min=input_data.start_time.isoformat(),
                time_max=end_time.isoformat(),
                max_results=input_data.max_events,
                single_events=True,
                search_term=input_data.search_term,
                show_deleted=False,
                show_hidden=input_data.include_declined_events,
                page_token=input_data.page_token,
            )

            # Format events into a user-friendly structure
            formatted_events = self._format_events(result.get("items", []))

            # Include next page token if available
            if next_page_token := result.get("nextPageToken"):
                yield "next_page_token", next_page_token

            for event in formatted_events:
                yield "event", event

            yield "events", formatted_events

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
            client_id=settings.secrets.google_client_id,
            client_secret=settings.secrets.google_client_secret,
            scopes=credentials.scopes,
        )
        return build("calendar", "v3", credentials=creds)

    def _read_calendar(
        self,
        service,
        calendarId: str,
        time_min: str,
        time_max: str,
        max_results: int,
        single_events: bool,
        search_term: str | None = None,
        show_deleted: bool = False,
        show_hidden: bool = False,
        page_token: str | None = None,
    ) -> dict:
        """Read calendar events with optional filtering."""
        calendar = service.events()

        # Build query parameters
        params = {
            "calendarId": calendarId,
            "timeMin": time_min,
            "timeMax": time_max,
            "maxResults": max_results,
            "singleEvents": single_events,
            "orderBy": "startTime",
            "showDeleted": show_deleted,
            "showHiddenInvitations": show_hidden,
            **({"pageToken": page_token} if page_token else {}),
        }

        # Add search term if provided
        if search_term:
            params["q"] = search_term

        result = calendar.list(**params).execute()
        return result

    def _format_events(self, events: list[dict]) -> list[CalendarEvent]:
        """Format Google Calendar API events into user-friendly structure."""
        formatted_events = []

        for event in events:
            # Determine if all-day event
            is_all_day = "date" in event.get("start", {})

            # Format start and end times
            if is_all_day:
                start_time = event.get("start", {}).get("date", "")
                end_time = event.get("end", {}).get("date", "")
            else:
                # Convert ISO format to more readable format
                start_datetime = datetime.fromisoformat(
                    event.get("start", {}).get("dateTime", "").replace("Z", "+00:00")
                )
                end_datetime = datetime.fromisoformat(
                    event.get("end", {}).get("dateTime", "").replace("Z", "+00:00")
                )
                start_time = start_datetime.strftime("%Y-%m-%d %H:%M")
                end_time = end_datetime.strftime("%Y-%m-%d %H:%M")

            # Extract attendees
            attendees = []
            for attendee in event.get("attendees", []):
                if email := attendee.get("email"):
                    attendees.append(email)

            # Check for video call link
            has_video_call = False
            video_link = None
            if conf_data := event.get("conferenceData"):
                if conf_url := conf_data.get("conferenceUrl"):
                    has_video_call = True
                    video_link = conf_url
                elif entry_points := conf_data.get("entryPoints", []):
                    for entry in entry_points:
                        if entry.get("entryPointType") == "video":
                            has_video_call = True
                            video_link = entry.get("uri")
                            break

            # Create formatted event
            formatted_event = CalendarEvent(
                id=event.get("id", ""),
                title=event.get("summary", "Untitled Event"),
                start_time=start_time,
                end_time=end_time,
                is_all_day=is_all_day,
                location=event.get("location"),
                description=event.get("description"),
                organizer=event.get("organizer", {}).get("email"),
                attendees=attendees,
                has_video_call=has_video_call,
                video_link=video_link,
                calendar_link=event.get("htmlLink", ""),
                is_recurring=bool(event.get("recurrence")),
            )

            formatted_events.append(formatted_event)

        return formatted_events


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
            default_factory=lambda: DurationTiming(
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
            default_factory=lambda: OneTimeEvent(discriminator="one_time"),
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
        super().__init__(
            id="ed2ec950-fbff-4204-94c0-023fb1d625e0",
            description="This block creates a new event in Google Calendar with customizable parameters.",
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarCreateEventBlock.Input,
            output_schema=GoogleCalendarCreateEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_title": "Team Meeting",
                "location": "Conference Room A",
                "description": "Weekly team sync-up",
                "calendar_id": "primary",
                "guest_emails": ["colleague1@example.com", "colleague2@example.com"],
                "add_google_meet": True,
                "send_notifications": True,
                "reminder_minutes": [
                    ReminderPreset.TEN_MINUTES.value,
                    ReminderPreset.ONE_HOUR.value,
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

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        try:
            service = self._build_service(credentials, **kwargs)

            # Create event body
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
            result = await asyncio.to_thread(
                self._create_event,
                service=service,
                calendar_id=input_data.calendar_id,
                event_body=event_body,
                send_notifications=input_data.send_notifications,
                conference_data_version=1 if input_data.add_google_meet else 0,
            )

            yield "event_id", result["id"]
            yield "event_link", result["htmlLink"]

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
            client_id=settings.secrets.google_client_id,
            client_secret=settings.secrets.google_client_secret,
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
