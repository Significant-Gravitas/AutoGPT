"""Unit tests for the Calendar read blocks and the shared Calendar helpers."""

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._calendar_api import (
    TEST_EVENT_RESOURCE,
    CalendarEvent,
    CalendarGuest,
    SendUpdates,
    calendar_error,
    format_calendar_event,
    resolve_time_zone,
    to_calendar_info,
    to_event_details,
)
from backend.blocks.google.calendar import GoogleCalendarReadEventsBlock
from backend.blocks.google.calendar_search import (
    GoogleCalendarGetEventBlock,
    GoogleCalendarListCalendarsBlock,
    GoogleCalendarSearchEventsBlock,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError


def test_read_events_formatting_is_unchanged():
    """Read Events now uses the shared formatter; pin what it produced before."""
    events = [
        {
            "id": "a",
            "summary": "Standup",
            "start": {"dateTime": "2026-10-06T09:00:00Z"},
            "end": {"dateTime": "2026-10-06T09:15:00-04:00"},
            "recurrence": ["RRULE:FREQ=DAILY"],
            "conferenceData": {
                "conferenceUrl": "https://meet.example/x",
                "entryPoints": [{"entryPointType": "video", "uri": "https://other"}],
            },
            "htmlLink": "https://calendar.example/a",
            "organizer": {"email": "boss@example.com"},
            "attendees": [{"email": "a@example.com"}, {"displayName": "No email"}],
        },
        {"id": "b", "start": {"date": "2026-10-07"}, "end": {"date": "2026-10-08"}},
    ]
    assert GoogleCalendarReadEventsBlock()._format_events(events) == [
        CalendarEvent(
            id="a",
            title="Standup",
            start_time="2026-10-06 09:00",
            end_time="2026-10-06 09:15",
            is_all_day=False,
            location=None,
            description=None,
            organizer="boss@example.com",
            attendees=["a@example.com"],
            has_video_call=True,
            video_link="https://meet.example/x",
            calendar_link="https://calendar.example/a",
            is_recurring=True,
        ),
        CalendarEvent(
            id="b",
            title="Untitled Event",
            start_time="2026-10-07",
            end_time="2026-10-08",
            is_all_day=True,
            location=None,
            description=None,
            organizer=None,
            attendees=[],
            has_video_call=False,
            video_link=None,
            calendar_link="",
            is_recurring=False,
        ),
    ]


def test_event_details_add_exact_times_guests_and_my_reply():
    occurrence = {**TEST_EVENT_RESOURCE, "recurringEventId": "series1"}
    details = to_event_details(occurrence, "team@group.calendar.google.com")
    base = format_calendar_event(occurrence).model_dump(exclude={"is_recurring"})
    assert details.model_dump(include=set(base)) == base
    assert details.calendar_id == "team@group.calendar.google.com"
    assert (details.start, details.end) == (
        "2026-10-06T15:00:00+01:00",
        "2026-10-06T16:00:00+01:00",
    )
    assert details.time_zone == "Europe/London"
    assert details.is_recurring is True
    assert details.recurring_event_id == "series1"
    assert details.my_response == "accepted"
    assert details.guests == [
        CalendarGuest(
            email="me@example.com", response="accepted", organizer=True, is_me=True
        ),
        CalendarGuest(email="alex@example.com", name="Alex Doe"),
    ]


def test_event_details_for_an_all_day_event_i_am_not_invited_to():
    details = to_event_details(
        {"id": "x", "start": {"date": "2026-10-07"}, "end": {"date": "2026-10-08"}},
        "primary",
    )
    assert (details.start, details.end, details.is_all_day) == (
        "2026-10-07",
        "2026-10-08",
        True,
    )
    assert (details.my_response, details.guests, details.status) == (
        None,
        [],
        "confirmed",
    )


def test_calendar_info_prefers_the_users_own_name_for_a_calendar():
    info = to_calendar_info(
        {"id": "x@group", "summary": "Team", "summaryOverride": "My team"}
    )
    assert (info.name, info.primary, info.access_role) == ("My team", False, "")


class _CalendarList:
    def __init__(self, pages: list[dict]):
        self.pages = pages
        self.calls: list[dict] = []

    def calendarList(self):
        return self

    def list(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self

    def execute(self):
        return self.pages[len(self.calls) - 1]


def test_list_calendars_reads_every_page():
    service = _CalendarList(
        [{"items": [{"id": "a"}], "nextPageToken": "p2"}, {"items": [{"id": "b"}]}]
    )
    items = GoogleCalendarListCalendarsBlock._list_calendars(
        service, writable_only=True, include_hidden=False
    )
    assert items == [{"id": "a"}, {"id": "b"}]
    assert service.calls == [
        {"maxResults": 250, "showHidden": False, "minAccessRole": "writer"},
        {
            "maxResults": 250,
            "showHidden": False,
            "minAccessRole": "writer",
            "pageToken": "p2",
        },
    ]


async def _outputs(block, input_data, context: ExecutionContext | None = None):
    return [
        output
        async for output in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=context or ExecutionContext(),
        )
    ]


@pytest.mark.asyncio
async def test_list_calendars_filters_by_name_or_id_with_the_main_one_first():
    block = GoogleCalendarListCalendarsBlock()
    block._list_calendars = lambda *args, **kwargs: [
        {
            "id": "c1@group.calendar.google.com",
            "summary": "Ops",
            "accessRole": "reader",
        },
        {
            "id": "me@example.com",
            "summary": "Me",
            "accessRole": "owner",
            "primary": True,
        },
    ]

    def run(**fields):
        return _outputs(
            block,
            GoogleCalendarListCalendarsBlock.Input.model_validate(
                {"credentials": TEST_CREDENTIALS_INPUT, **fields}
            ),
        )

    everything = dict(await run())["calendars"]
    assert [c.id for c in everything] == [
        "me@example.com",
        "c1@group.calendar.google.com",
    ]
    by_id = dict(await run(name_contains="GROUP.calendar"))["calendars"]
    assert [c.name for c in by_id] == ["Ops"]


def _search_input(**fields) -> GoogleCalendarSearchEventsBlock.Input:
    return GoogleCalendarSearchEventsBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


def test_search_covers_all_time_by_default():
    params = GoogleCalendarSearchEventsBlock()._search_params(
        _search_input(query="  planning "), "UTC"
    )
    assert params == {
        "calendarId": "primary",
        "q": "planning",
        "maxResults": 25,
        "singleEvents": False,
    }


def test_search_window_and_occurrences():
    params = GoogleCalendarSearchEventsBlock()._search_params(
        _search_input(
            query="review",
            calendar_id="team@example.com",
            expand_recurring=True,
            after="2026-10-01T09:00",
            before="2026-10-31T00:00:00Z",
            page_token="page-2",
        ),
        "America/New_York",
    )
    assert params == {
        "calendarId": "team@example.com",
        "q": "review",
        "maxResults": 25,
        "singleEvents": True,
        "orderBy": "startTime",
        "timeMin": "2026-10-01T13:00:00+00:00",
        "timeMax": "2026-10-31T00:00:00+00:00",
        "pageToken": "page-2",
    }


def test_search_needs_a_query_and_an_ordered_window():
    block = GoogleCalendarSearchEventsBlock()
    with pytest.raises(BlockInputError, match="Read Events"):
        block._search_params(_search_input(query="   "), "UTC")
    with pytest.raises(BlockInputError, match="later than"):
        block._search_params(
            _search_input(
                query="x", after="2026-10-02T00:00:00Z", before="2026-10-01T00:00:00Z"
            ),
            "UTC",
        )


def _get_input(event_id: str) -> GoogleCalendarGetEventBlock.Input:
    return GoogleCalendarGetEventBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "event_id": event_id}
    )


@pytest.mark.asyncio
async def test_get_event_reports_deleted_events():
    block = GoogleCalendarGetEventBlock()
    block._get_event = lambda *args: {**TEST_EVENT_RESOURCE, "status": "cancelled"}
    with pytest.raises(BlockExecutionError, match="deleted"):
        await _outputs(block, _get_input("evt1"))


@pytest.mark.asyncio
async def test_get_event_needs_an_id():
    with pytest.raises(BlockInputError, match="ID of the event"):
        await _outputs(GoogleCalendarGetEventBlock(), _get_input(" "))


@pytest.mark.parametrize(
    "status, reason, expected",
    [
        (404, "Not Found", "couldn't find that event or calendar"),
        (410, "Resource has been deleted", "already been deleted"),
        (412, "Precondition Failed", "changed while this block"),
        (403, "Request had insufficient authentication scopes.", "Reconnect Google"),
        (403, "You need to have writer access to this calendar.", "writer access"),
        (500, "Backend Error", "Google Calendar API error 500"),
    ],
)
def test_calendar_error_messages(status: int, reason: str, expected: str):
    content = f'{{"error": {{"code": {status}, "message": "{reason}"}}}}'.encode()
    exc = HttpError(httplib2.Response({"status": status}), content)
    assert expected in str(calendar_error(exc, "block", "id"))


def test_time_zone_comes_from_the_input_then_the_profile_then_utc():
    assert resolve_time_zone("Europe/London", "UTC", "b", "i").key == "Europe/London"
    assert (
        resolve_time_zone(" ", "America/New_York", "b", "i").key == "America/New_York"
    )
    assert resolve_time_zone("", "Not/A_Zone", "b", "i").key == "UTC"
    with pytest.raises(BlockInputError, match="recognised time zone"):
        resolve_time_zone("Not/A_Zone", "UTC", "b", "i")


def test_send_updates_maps_to_the_api_values():
    assert [option.api_value for option in SendUpdates] == [
        "all",
        "externalOnly",
        "none",
    ]
