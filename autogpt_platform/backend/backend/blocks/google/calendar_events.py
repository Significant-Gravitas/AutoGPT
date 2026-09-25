import asyncio
import uuid
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

from googleapiclient.errors import HttpError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField
from backend.util.exceptions import BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._calendar_api import (
    CALENDAR_EVENTS_SCOPE,
    CALENDAR_ID_DESCRIPTION,
    EVENT_ID_DESCRIPTION,
    TEST_EVENT_RESOURCE,
    CalendarEventDetails,
    SendUpdates,
    build_calendar_service,
    calendar_error,
    get_event,
    in_zone,
    require_event_id,
    resolve_time_zone,
    to_event_details,
)

_TEST_UPDATED_RESOURCE = {
    **TEST_EVENT_RESOURCE,
    "etag": '"3456789012399000"',
    "summary": "Quarterly planning (moved)",
    "start": {"dateTime": "2026-10-07T15:00:00+01:00", "timeZone": "Europe/London"},
    "end": {"dateTime": "2026-10-07T16:00:00+01:00", "timeZone": "Europe/London"},
    "attendees": [
        *TEST_EVENT_RESOURCE["attendees"],
        {"email": "sam@example.com", "responseStatus": "needsAction"},
    ],
}


class GoogleCalendarUpdateEventBlock(Block):
    """Change an existing Google Calendar event, keeping what isn't set."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_EVENTS_SCOPE]
        )
        event_id: str = SchemaField(description=EVENT_ID_DESCRIPTION)
        calendar_id: str = SchemaField(
            description=CALENDAR_ID_DESCRIPTION, default="primary"
        )
        title: str = SchemaField(
            description="New title. Empty keeps the current title.", default=""
        )
        start_time: datetime | None = SchemaField(
            description="New start time. Changing only the start keeps the event's length.",
            default=None,
        )
        end_time: datetime | None = SchemaField(
            description="New end time", default=None
        )
        time_zone: str = SchemaField(
            description="Time zone for new times given without a UTC offset, e.g. Europe/London. Empty uses your profile time zone.",
            default="",
            advanced=True,
        )
        location: str = SchemaField(
            description="New location. Empty keeps the current location.", default=""
        )
        description: str = SchemaField(
            description="New description. Empty keeps the current description.",
            default="",
        )
        add_guest_emails: list[str] = SchemaField(
            description="Email addresses to invite", default_factory=list
        )
        remove_guest_emails: list[str] = SchemaField(
            description="Email addresses to take off the guest list",
            default_factory=list,
        )
        add_google_meet: bool = SchemaField(
            description="Add a Google Meet link if the event doesn't have one",
            default=False,
        )
        send_updates: SendUpdates = SchemaField(
            description="Who Google emails about the change", default=SendUpdates.ALL
        )

    class Output(BlockSchemaOutput):
        event: CalendarEventDetails = SchemaField(
            description="The event after the change"
        )

    def __init__(self):
        super().__init__(
            id="0d1bdc92-075e-4452-b6f0-49536666f217",
            description=(
                "Change a Google Calendar event's title, time, location, "
                "description, guests or Google Meet link. Only the fields you set "
                "change, and guests can be emailed about the update."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarUpdateEventBlock.Input,
            output_schema=GoogleCalendarUpdateEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_id": TEST_EVENT_RESOURCE["id"],
                "title": "Quarterly planning (moved)",
                "start_time": "2026-10-07T15:00:00+01:00",
                "add_guest_emails": ["sam@example.com"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("event", to_event_details(_TEST_UPDATED_RESOURCE, "primary"))
            ],
            test_mock={
                "_get_event": lambda *args, **kwargs: TEST_EVENT_RESOURCE,
                "_patch_event": lambda *args, **kwargs: _TEST_UPDATED_RESOURCE,
            },
            is_irreversible_action=True,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        event_id = require_event_id(input_data.event_id, self.name, self.id)
        self._require_a_change(input_data)
        zone = resolve_time_zone(
            input_data.time_zone, execution_context.user_timezone, self.name, self.id
        )
        service = build_calendar_service(credentials)
        try:
            event = await asyncio.to_thread(
                self._get_event, service, input_data.calendar_id, event_id
            )
            body = self._patch_body(event, input_data, zone)
            if body:
                event = await asyncio.to_thread(
                    self._patch_event,
                    service,
                    input_data.calendar_id,
                    event_id,
                    body,
                    event.get("etag"),
                    input_data.send_updates.api_value,
                )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e
        yield "event", to_event_details(event, input_data.calendar_id)

    def _require_a_change(self, input_data: Input) -> None:
        if not (
            input_data.title
            or input_data.start_time
            or input_data.end_time
            or input_data.location
            or input_data.description
            or input_data.add_guest_emails
            or input_data.remove_guest_emails
            or input_data.add_google_meet
        ):
            raise self._input_error(
                "Nothing to change: set a new title, time, location, description, "
                "guests or Google Meet link."
            )

    def _patch_body(self, current: dict, input_data: Input, zone: tzinfo) -> dict:
        """The fields to send to events.patch; empty when nothing would change."""
        body: dict[str, Any] = {}
        if input_data.title:
            body["summary"] = input_data.title
        if input_data.location:
            body["location"] = input_data.location
        if input_data.description:
            body["description"] = input_data.description
        body.update(self._new_times(current, input_data, zone))
        guests = self._new_guests(
            current, input_data.add_guest_emails, input_data.remove_guest_emails
        )
        if guests is not None:
            body["attendees"] = guests
        if input_data.add_google_meet and not (
            current.get("conferenceData") or current.get("hangoutLink")
        ):
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": f"meet-{uuid.uuid4()}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        return body

    def _new_times(self, current: dict, input_data: Input, zone: tzinfo) -> dict:
        """New start and end. Moving only the start keeps the event's length."""
        given_start, given_end = input_data.start_time, input_data.end_time
        if given_start is None and given_end is None:
            return {}
        start = in_zone(given_start, zone) if given_start else None
        end = in_zone(given_end, zone) if given_end else None
        if start is None or end is None:
            old_start = _event_time(current.get("start", {}))
            old_end = _event_time(current.get("end", {}))
            if old_start is None or old_end is None:
                raise self._input_error(
                    "This is an all-day event, so give both a new start and a new end time."
                )
            if start is None:
                start = old_start
            if end is None:
                end = _shift(start, old_end - old_start)
        # Compare instants: datetimes sharing a ZoneInfo compare by wall clock.
        if end.astimezone(timezone.utc) <= start.astimezone(timezone.utc):
            raise self._input_error("The event must end after it starts.")
        # Times given without an offset were read in `zone`; make it the event's zone.
        naive = any(
            t is not None and t.tzinfo is None for t in (given_start, given_end)
        )
        zone_name = zone.key if naive and isinstance(zone, ZoneInfo) else None
        times = {"end": _time_body(end, zone_name)}
        if given_start is not None:
            times["start"] = _time_body(start, zone_name)
        return times

    def _new_guests(
        self, current: dict, add: list[str], remove: list[str]
    ) -> list[dict] | None:
        """The whole new guest list (patch replaces it), or None if unchanged."""
        if not add and not remove:
            return None
        if current.get("attendeesOmitted"):
            raise self._input_error(
                "Google didn't return this event's full guest list, so this "
                "account can't change the guests without dropping some."
            )
        existing = current.get("attendees", [])
        removed = {email.strip().lower() for email in remove if email.strip()}
        guests = [a for a in existing if a.get("email", "").lower() not in removed]
        known = {a.get("email", "").lower() for a in guests}
        for email in (email.strip() for email in add):
            if not email or email.lower() in known:
                continue
            if "@" not in email:
                raise self._input_error(f"'{email}' isn't an email address.")
            guests.append({"email": email})
            known.add(email.lower())
        return None if guests == existing else guests

    def _input_error(self, message: str) -> BlockInputError:
        return BlockInputError(message=message, block_name=self.name, block_id=self.id)

    @staticmethod
    def _get_event(service, calendar_id: str, event_id: str) -> dict:
        return get_event(service, calendar_id, event_id)

    @staticmethod
    def _patch_event(
        service,
        calendar_id: str,
        event_id: str,
        body: dict,
        etag: str | None,
        send_updates: str,
    ) -> dict:
        request = service.events().patch(
            calendarId=calendar_id,
            eventId=event_id,
            body=body,
            sendUpdates=send_updates,
            conferenceDataVersion=1 if "conferenceData" in body else 0,
        )
        if etag:
            # Google answers 412 if the event changed after we read it.
            request.headers["If-Match"] = etag
        return request.execute()


class GoogleCalendarDeleteEventBlock(Block):
    """Delete a Google Calendar event."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_EVENTS_SCOPE]
        )
        event_id: str = SchemaField(description=EVENT_ID_DESCRIPTION)
        calendar_id: str = SchemaField(
            description=CALENDAR_ID_DESCRIPTION, default="primary"
        )
        send_updates: SendUpdates = SchemaField(
            description="Who Google emails a cancellation to", default=SendUpdates.ALL
        )

    class Output(BlockSchemaOutput):
        event_id: str = SchemaField(description="ID of the deleted event")

    def __init__(self):
        super().__init__(
            id="5b74f483-fb7e-409e-87e7-5e1bcddb753b",
            description=(
                "Delete a Google Calendar event, or one occurrence of a repeating "
                "event, and optionally email the guests a cancellation."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarDeleteEventBlock.Input,
            output_schema=GoogleCalendarDeleteEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_id": TEST_EVENT_RESOURCE["id"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("event_id", TEST_EVENT_RESOURCE["id"])],
            test_mock={"_delete_event": lambda *args, **kwargs: None},
            is_irreversible_action=True,
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        event_id = require_event_id(input_data.event_id, self.name, self.id)
        service = build_calendar_service(credentials)
        try:
            await asyncio.to_thread(
                self._delete_event,
                service,
                input_data.calendar_id,
                event_id,
                input_data.send_updates.api_value,
            )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e
        yield "event_id", event_id

    @staticmethod
    def _delete_event(
        service, calendar_id: str, event_id: str, send_updates: str
    ) -> None:
        service.events().delete(
            calendarId=calendar_id, eventId=event_id, sendUpdates=send_updates
        ).execute()


def _event_time(value: dict) -> datetime | None:
    """A timed event's start or end; None for all-day events."""
    raw = value.get("dateTime")
    return datetime.fromisoformat(raw) if raw else None


def _shift(moment: datetime, by: timedelta) -> datetime:
    """Add a duration in UTC, so a daylight-saving change can't skew it."""
    return (moment.astimezone(timezone.utc) + by).astimezone(moment.tzinfo)


def _time_body(moment: datetime, zone_name: str | None) -> dict:
    """An event start/end. Clears ``date`` in case the event was all-day."""
    body: dict[str, Any] = {"dateTime": moment.isoformat(), "date": None}
    if zone_name:
        body["timeZone"] = zone_name
    return body
