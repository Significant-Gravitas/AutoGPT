"""Shared models and helpers for the Google Calendar blocks."""

from datetime import datetime, tzinfo
from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from backend.util.exceptions import BlockExecutionError, BlockInputError
from backend.util.settings import Settings

from ._auth import GoogleCredentials

CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"

EVENT_ID_DESCRIPTION = "ID of the event, e.g. from Search Events or Get Event"
CALENDAR_ID_DESCRIPTION = (
    "Calendar the event is on: 'primary' for the main calendar, or an ID from "
    "List Calendars"
)


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


class CalendarGuest(BaseModel):
    """Someone invited to a Calendar event, and their reply."""

    email: str = Field(description="The guest's email address")
    name: Optional[str] = Field(default=None, description="The guest's display name")
    response: str = Field(
        default="needsAction",
        description="Their reply: needsAction, accepted, declined or tentative",
    )
    comment: Optional[str] = Field(
        default=None, description="The note they added to their reply"
    )
    optional: bool = Field(default=False, description="Whether they are optional")
    organizer: bool = Field(default=False, description="Whether they organize it")
    is_me: bool = Field(
        default=False, description="Whether this is the connected Google account"
    )


class CalendarEventDetails(CalendarEvent):
    """A CalendarEvent with exact times, status and every guest's reply."""

    calendar_id: str = Field(description="Calendar the event was read from")
    start: str = Field(
        description="Exact start: date-time with UTC offset, or YYYY-MM-DD for all-day events"
    )
    end: str = Field(
        description="Exact end: date-time with UTC offset, or YYYY-MM-DD (exclusive) for all-day events"
    )
    time_zone: Optional[str] = Field(
        default=None, description="The event's own time zone, when it has one"
    )
    status: str = Field(
        default="confirmed", description="confirmed, tentative or cancelled"
    )
    guests: list[CalendarGuest] = Field(
        default_factory=list, description="Everyone invited, with their replies"
    )
    my_response: Optional[str] = Field(
        default=None,
        description="Your reply (needsAction, accepted, declined or tentative); empty when you aren't a guest",
    )
    recurring_event_id: Optional[str] = Field(
        default=None,
        description="ID of the repeating event this is one occurrence of",
    )


class CalendarInfo(BaseModel):
    """A calendar on the user's Google Calendar list."""

    id: str = Field(description="Calendar ID to pass to the other Calendar blocks")
    name: str = Field(description="The calendar's name as the user sees it")
    description: Optional[str] = Field(
        default=None, description="The calendar's description"
    )
    time_zone: Optional[str] = Field(
        default=None, description="The calendar's time zone (IANA name)"
    )
    access_role: str = Field(
        description="The user's access: owner, writer, reader or freeBusyReader"
    )
    primary: bool = Field(
        default=False, description="Whether this is the user's main calendar"
    )


class SendUpdates(str, Enum):
    """Who Google emails about a change to an event."""

    ALL = "all"
    EXTERNAL_ONLY = "external_only"
    NONE = "none"

    @property
    def api_value(self) -> str:
        return "externalOnly" if self is SendUpdates.EXTERNAL_ONLY else self.value


TEST_EVENT_RESOURCE: dict[str, Any] = {
    "id": "7kq2m1v9c3b8d4e5f6g7h8i9j0",
    "etag": '"3456789012345000"',
    "status": "confirmed",
    "htmlLink": "https://www.google.com/calendar/event?eid=N2txMm0xdjljM2I4ZDRlNWY2ZzdoOGk5ajA",
    "summary": "Quarterly planning",
    "description": "Roadmap and hiring",
    "location": "Room 4B",
    "start": {"dateTime": "2026-10-06T15:00:00+01:00", "timeZone": "Europe/London"},
    "end": {"dateTime": "2026-10-06T16:00:00+01:00", "timeZone": "Europe/London"},
    "organizer": {"email": "me@example.com", "self": True},
    "attendees": [
        {
            "email": "me@example.com",
            "organizer": True,
            "self": True,
            "responseStatus": "accepted",
        },
        {
            "email": "alex@example.com",
            "displayName": "Alex Doe",
            "responseStatus": "needsAction",
        },
    ],
    "hangoutLink": "https://meet.google.com/abc-defg-hij",
    "conferenceData": {
        "entryPoints": [
            {"entryPointType": "video", "uri": "https://meet.google.com/abc-defg-hij"}
        ]
    },
}


def build_calendar_service(credentials: GoogleCredentials):
    settings = Settings()
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
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def get_event(service, calendar_id: str, event_id: str) -> dict:
    return service.events().get(calendarId=calendar_id, eventId=event_id).execute()


def format_calendar_event(event: dict) -> CalendarEvent:
    """Format a Google Calendar API event into a CalendarEvent."""
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
    return CalendarEvent(
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


def to_event_details(event: dict[str, Any], calendar_id: str) -> CalendarEventDetails:
    """Map a Calendar API event resource to CalendarEventDetails."""
    fields = format_calendar_event(event).model_dump()
    # Occurrences of a repeating event carry recurringEventId, not recurrence.
    fields["is_recurring"] = bool(
        event.get("recurrence") or event.get("recurringEventId")
    )
    attendees = [a for a in event.get("attendees", []) if a.get("email")]
    me = next((a for a in attendees if a.get("self")), {})
    start, end = event.get("start", {}), event.get("end", {})
    return CalendarEventDetails(
        **fields,
        calendar_id=calendar_id,
        start=start.get("dateTime") or start.get("date", ""),
        end=end.get("dateTime") or end.get("date", ""),
        time_zone=start.get("timeZone"),
        status=event.get("status", "confirmed"),
        guests=[
            CalendarGuest(
                email=a["email"],
                name=a.get("displayName"),
                response=a.get("responseStatus", "needsAction"),
                comment=a.get("comment"),
                optional=bool(a.get("optional")),
                organizer=bool(a.get("organizer")),
                is_me=bool(a.get("self")),
            )
            for a in attendees
        ],
        my_response=me.get("responseStatus"),
        recurring_event_id=event.get("recurringEventId"),
    )


def to_calendar_info(item: dict[str, Any]) -> CalendarInfo:
    """Map a calendarList entry to CalendarInfo."""
    return CalendarInfo(
        id=item["id"],
        name=item.get("summaryOverride") or item.get("summary", ""),
        description=item.get("description"),
        time_zone=item.get("timeZone"),
        access_role=item.get("accessRole", ""),
        primary=bool(item.get("primary")),
    )


def resolve_time_zone(
    name: str, profile_time_zone: str, block_name: str, block_id: str
) -> ZoneInfo:
    """The block's time_zone input, else the user's profile time zone, else UTC."""
    if name.strip():
        try:
            return ZoneInfo(name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            raise BlockInputError(
                message=(
                    f"'{name}' isn't a recognised time zone. Use an IANA name "
                    "such as Europe/London or America/New_York."
                ),
                block_name=block_name,
                block_id=block_id,
            )
    try:
        return ZoneInfo(profile_time_zone or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def require_event_id(event_id: str, block_name: str, block_id: str) -> str:
    """An empty ID would turn events.get into a request for the event list."""
    if not event_id.strip():
        raise BlockInputError(
            message="Give the ID of the event, e.g. from Search Events or Get Event.",
            block_name=block_name,
            block_id=block_id,
        )
    return event_id.strip()


def in_zone(moment: datetime, zone: tzinfo) -> datetime:
    """Read a time without a UTC offset as local time in ``zone``."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=zone)


def calendar_error(
    exc: HttpError, block_name: str, block_id: str
) -> BlockExecutionError:
    """Turn a Calendar API error into a message the user can act on."""
    if exc.status_code == 404:
        message = (
            "Google Calendar couldn't find that event or calendar, or the "
            "connected Google account can't see it."
        )
    elif exc.status_code == 410:
        message = "That event has already been deleted from Google Calendar."
    elif exc.status_code == 412:
        message = (
            "The event changed while this block was updating it. Run the block "
            "again to apply your changes to the latest version."
        )
    elif exc.status_code == 403 and "insufficient" in str(exc.reason).lower():
        message = (
            "The connected Google account hasn't granted the Calendar access "
            "this block needs. Reconnect Google and approve Calendar access."
        )
    else:
        message = f"Google Calendar API error {exc.status_code}: {exc.reason}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )
