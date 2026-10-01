import asyncio
from datetime import datetime, timezone, tzinfo
from typing import Any

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
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._auth import (
    GOOGLE_OAUTH_IS_CONFIGURED,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    GoogleCredentials,
    GoogleCredentialsField,
    GoogleCredentialsInput,
)
from ._calendar_api import (
    CALENDAR_ID_DESCRIPTION,
    CALENDAR_READONLY_SCOPE,
    TEST_EVENT_RESOURCE,
    CalendarEventDetails,
    CalendarInfo,
    build_calendar_service,
    calendar_error,
    get_event,
    in_zone,
    require_event_id,
    resolve_time_zone,
    to_calendar_info,
    to_event_details,
)

_TEST_CALENDARS = [
    {
        "id": "me@example.com",
        "summary": "me@example.com",
        "timeZone": "Europe/London",
        "accessRole": "owner",
        "primary": True,
    },
    {
        "id": "c_4f1d2e3a9b8c7d6e@group.calendar.google.com",
        "summary": "Ops rota",
        "description": "On-call shifts and releases",
        "timeZone": "Europe/London",
        "accessRole": "writer",
    },
    {
        "id": "en.uk#holiday@group.v.calendar.google.com",
        "summary": "Holidays in United Kingdom",
        "timeZone": "Europe/London",
        "accessRole": "reader",
    },
]
_TEST_EVENT = to_event_details(TEST_EVENT_RESOURCE, "primary")


class GoogleCalendarListCalendarsBlock(Block):
    """List the calendars on the user's Google Calendar list."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_READONLY_SCOPE]
        )
        name_contains: str = SchemaField(
            description="Only calendars whose name or ID contains this text (not case-sensitive)",
            default="",
        )
        writable_only: bool = SchemaField(
            description="Only calendars the account can add or change events on",
            default=False,
        )
        include_hidden: bool = SchemaField(
            description="Include calendars the user has hidden from their list",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        calendars: list[CalendarInfo] = SchemaField(
            description="Matching calendars, the main calendar first"
        )
        calendar: CalendarInfo = SchemaField(description="Each matching calendar")

    def __init__(self):
        ops_calendar = to_calendar_info(_TEST_CALENDARS[1])
        super().__init__(
            id="e4673d4a-6350-436b-b771-9ab7f2fde9b0",
            description=(
                "List the user's Google calendars with each one's ID, name, time "
                "zone and the user's access level. Use it to turn a calendar name "
                "into the calendar ID the other Calendar blocks need."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleCalendarListCalendarsBlock.Input,
            output_schema=GoogleCalendarListCalendarsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "name_contains": "ops"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("calendars", [ops_calendar]), ("calendar", ops_calendar)],
            test_mock={"_list_calendars": lambda *args, **kwargs: _TEST_CALENDARS},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        service = build_calendar_service(credentials)
        try:
            items = await asyncio.to_thread(
                self._list_calendars,
                service,
                writable_only=input_data.writable_only,
                include_hidden=input_data.include_hidden,
            )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e

        needle = input_data.name_contains.strip().lower()
        calendars = [
            calendar
            for calendar in map(to_calendar_info, items)
            if needle in calendar.name.lower() or needle in calendar.id.lower()
        ]
        calendars.sort(key=lambda calendar: not calendar.primary)
        yield "calendars", calendars
        for calendar in calendars:
            yield "calendar", calendar

    @staticmethod
    def _list_calendars(
        service, *, writable_only: bool, include_hidden: bool
    ) -> list[dict]:
        items: list[dict] = []
        params: dict[str, Any] = {"maxResults": 250, "showHidden": include_hidden}
        if writable_only:
            params["minAccessRole"] = "writer"
        while True:
            response = service.calendarList().list(**params).execute()
            items.extend(response.get("items", []))
            if not response.get("nextPageToken"):
                return items
            params["pageToken"] = response["nextPageToken"]


class GoogleCalendarSearchEventsBlock(Block):
    """Search a Google Calendar for events by keyword, past or future."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_READONLY_SCOPE]
        )
        query: str = SchemaField(
            description="Words to look for in event titles, descriptions, locations, and guests' names and emails"
        )
        calendar_id: str = SchemaField(
            description="Calendar to search: 'primary' for the main calendar, or an ID from List Calendars",
            default="primary",
        )
        after: datetime | None = SchemaField(
            description="Only events that end after this time. Empty includes past events. Times without a UTC offset use your profile time zone.",
            default=None,
        )
        before: datetime | None = SchemaField(
            description="Only events that start before this time. Empty means no limit.",
            default=None,
        )
        expand_recurring: bool = SchemaField(
            description="List every occurrence of repeating events, in start-time order. Off lists each repeating event once.",
            default=False,
        )
        max_results: int = SchemaField(
            description="Maximum number of events to return",
            default=25,
            ge=1,
            le=250,
        )
        page_token: str = SchemaField(
            description="Page token from a previous search, to get the next page",
            default="",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        events: list[CalendarEventDetails] = SchemaField(
            description="Matching events with their times, guests and replies"
        )
        event: CalendarEventDetails = SchemaField(description="Each matching event")
        next_page_token: str = SchemaField(
            description="Token for the next page, when there are more results"
        )

    def __init__(self):
        super().__init__(
            id="88f456a6-5b49-4ea2-a65e-9810ec65dcea",
            description=(
                "Search a Google Calendar for events by keyword, across past and "
                "future events, optionally within a time range. Returns each event "
                "with its time, location, video link, guests and their replies."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleCalendarSearchEventsBlock.Input,
            output_schema=GoogleCalendarSearchEventsBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "planning"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("events", [_TEST_EVENT]),
                ("event", _TEST_EVENT),
                ("next_page_token", "next-page"),
            ],
            test_mock={
                "_list_events": lambda *args, **kwargs: {
                    "items": [TEST_EVENT_RESOURCE],
                    "nextPageToken": "next-page",
                }
            },
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: GoogleCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        params = self._search_params(input_data, execution_context.user_timezone)
        service = build_calendar_service(credentials)
        try:
            result = await asyncio.to_thread(self._list_events, service, params)
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e

        events = [
            to_event_details(item, input_data.calendar_id)
            for item in result.get("items", [])
        ]
        yield "events", events
        for event in events:
            yield "event", event
        if next_page_token := result.get("nextPageToken"):
            yield "next_page_token", next_page_token

    def _search_params(self, input_data: Input, profile_time_zone: str) -> dict:
        """Build the events.list parameters for a search."""
        query = input_data.query.strip()
        if not query:
            raise BlockInputError(
                message=(
                    "Enter something to search for. To list events in a time "
                    "range, use Google Calendar Read Events."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        zone = resolve_time_zone("", profile_time_zone, self.name, self.id)
        params: dict[str, Any] = {
            "calendarId": input_data.calendar_id,
            "q": query,
            "maxResults": input_data.max_results,
            "singleEvents": input_data.expand_recurring,
        }
        if input_data.expand_recurring:
            params["orderBy"] = "startTime"
        after = _utc(input_data.after, zone) if input_data.after else None
        before = _utc(input_data.before, zone) if input_data.before else None
        if after and before and before <= after:
            raise BlockInputError(
                message="'before' must be later than 'after'.",
                block_name=self.name,
                block_id=self.id,
            )
        if after:
            params["timeMin"] = after.isoformat(timespec="seconds")
        if before:
            params["timeMax"] = before.isoformat(timespec="seconds")
        if input_data.page_token:
            params["pageToken"] = input_data.page_token
        return params

    @staticmethod
    def _list_events(service, params: dict) -> dict:
        return service.events().list(**params).execute()


class GoogleCalendarGetEventBlock(Block):
    """Get one Google Calendar event by its ID."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_READONLY_SCOPE]
        )
        event_id: str = SchemaField(
            description="ID of the event, e.g. from Search Events, Read Events or Create Event"
        )
        calendar_id: str = SchemaField(
            description=CALENDAR_ID_DESCRIPTION, default="primary"
        )

    class Output(BlockSchemaOutput):
        event: CalendarEventDetails = SchemaField(
            description="The event with its time, guests and their replies"
        )

    def __init__(self):
        super().__init__(
            id="cbbf32b5-0367-41e2-b090-1c85475944cb",
            description=(
                "Get one Google Calendar event by its ID: title, exact times, "
                "location, description, video link, organizer, guests and their replies."
            ),
            categories={BlockCategory.PRODUCTIVITY, BlockCategory.DATA},
            input_schema=GoogleCalendarGetEventBlock.Input,
            output_schema=GoogleCalendarGetEventBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "event_id": TEST_EVENT_RESOURCE["id"],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("event", _TEST_EVENT)],
            test_mock={"_get_event": lambda *args, **kwargs: TEST_EVENT_RESOURCE},
        )

    async def run(
        self, input_data: Input, *, credentials: GoogleCredentials, **kwargs
    ) -> BlockOutput:
        event_id = require_event_id(input_data.event_id, self.name, self.id)
        service = build_calendar_service(credentials)
        try:
            event = await asyncio.to_thread(
                self._get_event, service, input_data.calendar_id, event_id
            )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e
        if event.get("status") == "cancelled":
            raise BlockExecutionError(
                message="That event has been deleted from Google Calendar.",
                block_name=self.name,
                block_id=self.id,
            )
        yield "event", to_event_details(event, input_data.calendar_id)

    @staticmethod
    def _get_event(service, calendar_id: str, event_id: str) -> dict:
        return get_event(service, calendar_id, event_id)


def _utc(moment: datetime, zone: tzinfo) -> datetime:
    return in_zone(moment, zone).astimezone(timezone.utc)
