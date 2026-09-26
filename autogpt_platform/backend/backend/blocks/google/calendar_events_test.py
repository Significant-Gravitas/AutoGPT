"""Unit tests for the Update Event and Delete Event blocks.

The blocks' own test_mock cases skip the request building; these cover it.
"""

from zoneinfo import ZoneInfo

import httplib2
import pytest
from googleapiclient.errors import HttpError

from backend.blocks.google import calendar_events
from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._calendar_api import TEST_EVENT_RESOURCE
from backend.blocks.google.calendar_events import (
    GoogleCalendarDeleteEventBlock,
    GoogleCalendarUpdateEventBlock,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
EVENT = TEST_EVENT_RESOURCE


def http_error(status: int, reason: str) -> HttpError:
    content = f'{{"error": {{"code": {status}, "message": "{reason}"}}}}'.encode()
    return HttpError(httplib2.Response({"status": status}), content)


class _Request:
    def __init__(self, result=None, error: HttpError | None = None):
        self.headers: dict[str, str] = {}
        self._result = result
        self._error = error

    def execute(self):
        if self._error:
            raise self._error
        return self._result


def merge_patch(target: dict, patch: dict) -> dict:
    """Apply a patch the way Google does: null deletes, objects merge, lists replace."""
    merged = dict(target)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_patch(merged[key], value)
        else:
            merged[key] = value
    return merged


class _Events:
    """Fake events() resource. patch() merge-patches the body into the event."""

    def __init__(self, event: dict | None = None, error: HttpError | None = None):
        self.event = event or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []
        self.requests: list[_Request] = []

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return _Request(self.event)

    def patch(self, **kwargs):
        self.calls.append(("patch", kwargs))
        request = _Request(merge_patch(self.event, kwargs["body"]), self.error)
        self.requests.append(request)
        return request

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return _Request(error=self.error)


class _Service:
    def __init__(self, events: _Events):
        self._events = events

    def events(self):
        return self._events


def _update_input(**fields) -> GoogleCalendarUpdateEventBlock.Input:
    return GoogleCalendarUpdateEventBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "event_id": EVENT["id"], **fields}
    )


def _patch_body(event: dict, zone=LONDON, **fields) -> dict:
    return GoogleCalendarUpdateEventBlock()._patch_body(
        event, _update_input(**fields), zone
    )


def test_text_fields_change_only_when_set():
    assert _patch_body(EVENT, title="Renamed", location="Room 1") == {
        "summary": "Renamed",
        "location": "Room 1",
    }


def test_moving_only_the_start_keeps_the_length():
    assert _patch_body(EVENT, start_time="2026-10-07T09:30:00-04:00") == {
        "start": {"dateTime": "2026-10-07T09:30:00-04:00", "date": None},
        "end": {"dateTime": "2026-10-07T10:30:00-04:00", "date": None},
    }


def test_times_without_offset_use_the_zone_and_become_the_events_zone():
    body = _patch_body(EVENT, NEW_YORK, start_time="2026-10-07T09:30")
    assert body["start"] == {
        "dateTime": "2026-10-07T09:30:00-04:00",
        "date": None,
        "timeZone": "America/New_York",
    }
    assert body["end"]["dateTime"] == "2026-10-07T10:30:00-04:00"
    assert body["end"]["timeZone"] == "America/New_York"


def test_moving_only_the_end():
    assert _patch_body(EVENT, end_time="2026-10-06T17:00:00+01:00") == {
        "end": {"dateTime": "2026-10-06T17:00:00+01:00", "date": None}
    }


def test_the_event_must_end_after_it_starts():
    with pytest.raises(BlockInputError, match="end after it starts"):
        _patch_body(EVENT, end_time="2026-10-06T14:00:00+01:00")
    with pytest.raises(BlockInputError, match="end after it starts"):
        _patch_body(
            EVENT,
            start_time="2026-10-06T15:00:00Z",
            end_time="2026-10-06T15:00:00Z",
        )


def test_the_length_survives_a_daylight_saving_change():
    event = {
        **EVENT,
        "start": {"dateTime": "2026-10-31T12:00:00-04:00"},
        "end": {"dateTime": "2026-10-31T13:00:00-04:00"},
    }
    body = _patch_body(event, NEW_YORK, start_time="2026-11-01T01:30")
    assert body["start"]["dateTime"] == "2026-11-01T01:30:00-04:00"
    assert body["end"]["dateTime"] == "2026-11-01T01:30:00-05:00"


def test_all_day_events_need_both_times():
    all_day = {**EVENT, "start": {"date": "2026-10-06"}, "end": {"date": "2026-10-07"}}
    with pytest.raises(BlockInputError, match="all-day"):
        _patch_body(all_day, start_time="2026-10-06T09:00:00Z")
    body = _patch_body(
        all_day, start_time="2026-10-06T09:00:00Z", end_time="2026-10-06T10:00:00Z"
    )
    assert body["start"] == {"dateTime": "2026-10-06T09:00:00+00:00", "date": None}


def test_guest_changes_send_the_whole_list_and_keep_replies():
    event = {
        **EVENT,
        "attendees": [
            *EVENT["attendees"],
            {"email": "old@example.com", "responseStatus": "declined"},
        ],
    }
    body = _patch_body(
        event,
        add_guest_emails=["sam@example.com", "ALEX@example.com", " "],
        remove_guest_emails=["Old@Example.com"],
    )
    assert body == {
        "attendees": [*EVENT["attendees"], {"email": "sam@example.com"}],
    }


def test_guest_changes_that_change_nothing_send_nothing():
    body = _patch_body(
        EVENT,
        add_guest_emails=["alex@example.com"],
        remove_guest_emails=["nobody@example.com"],
    )
    assert body == {}


def test_guest_changes_refused_when_google_hid_part_of_the_list():
    with pytest.raises(BlockInputError, match="full guest list"):
        _patch_body(
            {**EVENT, "attendeesOmitted": True}, add_guest_emails=["sam@example.com"]
        )


def test_guest_emails_are_checked():
    with pytest.raises(BlockInputError, match="isn't an email"):
        _patch_body(EVENT, add_guest_emails=["sam"])


def test_meet_link_is_added_only_when_missing():
    no_meet = {
        key: value
        for key, value in EVENT.items()
        if key not in ("hangoutLink", "conferenceData")
    }
    request = _patch_body(no_meet, add_google_meet=True)["conferenceData"][
        "createRequest"
    ]
    assert request["conferenceSolutionKey"] == {"type": "hangoutsMeet"}
    assert request["requestId"].startswith("meet-")
    assert _patch_body(EVENT, add_google_meet=True) == {}


def test_update_needs_something_to_change():
    with pytest.raises(BlockInputError, match="Nothing to change"):
        GoogleCalendarUpdateEventBlock()._require_a_change(_update_input())


def test_patch_sends_the_etag_as_if_match():
    events = _Events(EVENT)
    GoogleCalendarUpdateEventBlock._patch_event(
        _Service(events), "primary", "evt1", {"summary": "x"}, '"123"', "externalOnly"
    )
    assert events.calls[-1] == (
        "patch",
        {
            "calendarId": "primary",
            "eventId": "evt1",
            "body": {"summary": "x"},
            "sendUpdates": "externalOnly",
            "conferenceDataVersion": 0,
        },
    )
    assert events.requests[-1].headers == {"If-Match": '"123"'}


def test_patch_asks_for_conference_data_when_adding_meet():
    events = _Events(EVENT)
    GoogleCalendarUpdateEventBlock._patch_event(
        _Service(events), "primary", "evt1", {"conferenceData": {}}, None, "all"
    )
    assert events.calls[-1][1]["conferenceDataVersion"] == 1
    assert events.requests[-1].headers == {}


async def _run(block, input_data, context: ExecutionContext | None = None):
    return [
        output
        async for output in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=context or ExecutionContext(),
        )
    ]


@pytest.fixture
def events(monkeypatch) -> _Events:
    fake = _Events(EVENT)
    monkeypatch.setattr(
        calendar_events, "build_calendar_service", lambda credentials: _Service(fake)
    )
    return fake


@pytest.mark.asyncio
async def test_update_reads_then_patches_the_version_it_read(events: _Events):
    outputs = await _run(
        GoogleCalendarUpdateEventBlock(),
        _update_input(title="Renamed", send_updates="external_only"),
    )
    assert [name for name, _ in events.calls] == ["get", "patch"]
    assert events.calls[1][1]["sendUpdates"] == "externalOnly"
    assert events.requests[-1].headers["If-Match"] == EVENT["etag"]
    assert outputs[0][1].title == "Renamed"


@pytest.mark.asyncio
async def test_update_skips_the_write_when_nothing_would_change(events: _Events):
    outputs = await _run(
        GoogleCalendarUpdateEventBlock(),
        _update_input(add_guest_emails=["alex@example.com"]),
    )
    assert [name for name, _ in events.calls] == ["get"]
    assert outputs[0][1].id == EVENT["id"]


@pytest.mark.asyncio
async def test_update_reads_offsetless_times_in_the_profile_time_zone(
    events: _Events,
):
    await _run(
        GoogleCalendarUpdateEventBlock(),
        _update_input(start_time="2026-10-07T09:00"),
        ExecutionContext(user_timezone="America/New_York"),
    )
    body = events.calls[-1][1]["body"]
    assert body["start"]["dateTime"] == "2026-10-07T09:00:00-04:00"
    assert body["start"]["timeZone"] == "America/New_York"


@pytest.mark.asyncio
async def test_update_explains_a_version_conflict(events: _Events):
    events.error = http_error(412, "Precondition Failed")
    with pytest.raises(BlockExecutionError, match="changed while"):
        await _run(GoogleCalendarUpdateEventBlock(), _update_input(title="Renamed"))


@pytest.mark.asyncio
async def test_a_blank_event_id_is_rejected_before_calling_google(events: _Events):
    with pytest.raises(BlockInputError, match="ID of the event"):
        await _run(
            GoogleCalendarDeleteEventBlock(),
            GoogleCalendarDeleteEventBlock.Input.model_validate(
                {"credentials": TEST_CREDENTIALS_INPUT, "event_id": "  "}
            ),
        )
    assert events.calls == []


def test_delete_passes_the_cancellation_setting():
    events = _Events()
    GoogleCalendarDeleteEventBlock._delete_event(
        _Service(events), "team@example.com", "evt1", "none"
    )
    assert events.calls == [
        (
            "delete",
            {
                "calendarId": "team@example.com",
                "eventId": "evt1",
                "sendUpdates": "none",
            },
        )
    ]


@pytest.mark.asyncio
async def test_delete_of_a_deleted_event_says_so(events: _Events):
    events.error = http_error(410, "Resource has been deleted")
    with pytest.raises(BlockExecutionError, match="already been deleted"):
        await _run(
            GoogleCalendarDeleteEventBlock(),
            GoogleCalendarDeleteEventBlock.Input.model_validate(
                {"credentials": TEST_CREDENTIALS_INPUT, "event_id": "evt1"}
            ),
        )
