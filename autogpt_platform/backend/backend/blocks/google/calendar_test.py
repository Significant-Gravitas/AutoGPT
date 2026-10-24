"""Regression tests for the Google Calendar Read Events and Create Event blocks."""

import pytest

from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google.calendar import (
    GoogleCalendarCreateEventBlock,
    GoogleCalendarReadEventsBlock,
)
from backend.data.execution import ExecutionContext


def _event(event_id: str, my_response: str) -> dict:
    return {
        "id": event_id,
        "summary": event_id.title(),
        "start": {"dateTime": "2026-10-05T09:00:00Z"},
        "end": {"dateTime": "2026-10-05T10:00:00Z"},
        "attendees": [
            {"email": "boss@example.com", "organizer": True},
            {"email": "me@example.com", "self": True, "responseStatus": my_response},
        ],
    }


class _FakeCalendar:
    """Stands in for the Calendar service and records what the block sends."""

    def __init__(self, items: list[dict] | None = None):
        self.items = items or []
        self.list_params: dict = {}
        self.insert_kwargs: dict = {}
        self.result: dict = {}

    def events(self):
        return self

    def list(self, **params):
        self.list_params = params
        self.result = {"items": self.items}
        return self

    def insert(self, **kwargs):
        self.insert_kwargs = kwargs
        self.result = {"id": "evt1", "htmlLink": "https://calendar.example/evt1"}
        return self

    def execute(self):
        return self.result


async def _run(block, service: _FakeCalendar, user_timezone: str = "UTC", **fields):
    block._build_service = lambda *args, **kwargs: service
    input_data = block.input_schema.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )
    outputs = [
        output
        async for output in block.run(
            input_data,
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(user_timezone=user_timezone),
        )
    ]
    assert [value for name, value in outputs if name == "error"] == []
    return outputs


@pytest.mark.asyncio
async def test_read_events_reads_times_without_an_offset_in_the_users_zone():
    service = _FakeCalendar()
    await _run(
        GoogleCalendarReadEventsBlock(),
        service,
        "America/New_York",
        start_time="2026-10-05T09:00",
        time_range_days=2,
    )
    assert service.list_params["timeMin"] == "2026-10-05T09:00:00-04:00"
    assert service.list_params["timeMax"] == "2026-10-07T09:00:00-04:00"


@pytest.mark.asyncio
async def test_read_events_keeps_an_explicit_offset():
    service = _FakeCalendar()
    await _run(
        GoogleCalendarReadEventsBlock(),
        service,
        "America/New_York",
        start_time="2026-10-05T09:00:00+00:00",
    )
    assert service.list_params["timeMin"] == "2026-10-05T09:00:00+00:00"


@pytest.mark.asyncio
async def test_read_events_falls_back_to_utc_for_an_unknown_profile_zone():
    service = _FakeCalendar()
    await _run(
        GoogleCalendarReadEventsBlock(),
        service,
        "Not/A_Zone",
        start_time="2026-10-05T09:00",
    )
    assert service.list_params["timeMin"] == "2026-10-05T09:00:00+00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "include_declined, expected_ids",
    [(False, ["going"]), (True, ["going", "skipping"])],
)
async def test_read_events_leaves_out_declined_events_unless_asked(
    include_declined: bool, expected_ids: list[str]
):
    service = _FakeCalendar(
        [_event("going", "accepted"), _event("skipping", "declined")]
    )
    outputs = await _run(
        GoogleCalendarReadEventsBlock(),
        service,
        start_time="2026-10-05T00:00:00Z",
        include_declined_events=include_declined,
    )
    assert [event.id for event in dict(outputs)["events"]] == expected_ids
    # Hidden invitations are a different thing from declined events.
    assert service.list_params.get("showHiddenInvitations", False) is False


def test_occurrences_of_a_repeating_event_are_recurring():
    occurrence = {
        "id": "series1_20261005T090000Z",
        "recurringEventId": "series1",
        "start": {"dateTime": "2026-10-05T09:00:00Z"},
        "end": {"dateTime": "2026-10-05T10:00:00Z"},
    }
    one_off = {**occurrence, "id": "one-off", "recurringEventId": None}
    events = GoogleCalendarReadEventsBlock()._format_events([occurrence, one_off])
    assert [event.is_recurring for event in events] == [True, False]


@pytest.mark.asyncio
async def test_create_event_reads_times_without_an_offset_in_the_users_zone():
    service = _FakeCalendar()
    outputs = await _run(
        GoogleCalendarCreateEventBlock(),
        service,
        "Europe/London",
        event_title="Planning",
        timing={
            "discriminator": "exact_timing",
            "start_datetime": "2026-10-06T15:00",
            "end_datetime": "2026-10-06T16:00",
        },
    )
    assert ("event_id", "evt1") in outputs
    body = service.insert_kwargs["body"]
    assert body["start"] == {
        "dateTime": "2026-10-06T15:00:00",
        "timeZone": "Europe/London",
    }
    assert body["end"] == {
        "dateTime": "2026-10-06T16:00:00",
        "timeZone": "Europe/London",
    }


@pytest.mark.asyncio
async def test_create_event_keeps_an_explicit_offset_for_one_off_events():
    service = _FakeCalendar()
    await _run(
        GoogleCalendarCreateEventBlock(),
        service,
        "Europe/London",
        event_title="Planning",
        timing={
            "discriminator": "duration_timing",
            "start_datetime": "2026-10-06T15:00:00+02:00",
            "duration_minutes": 30,
        },
    )
    body = service.insert_kwargs["body"]
    assert body["start"] == {"dateTime": "2026-10-06T15:00:00+02:00"}
    assert body["end"] == {"dateTime": "2026-10-06T15:30:00+02:00"}


@pytest.mark.asyncio
async def test_repeating_events_get_the_time_zone_google_requires():
    service = _FakeCalendar()
    await _run(
        GoogleCalendarCreateEventBlock(),
        service,
        "Europe/London",
        event_title="Standup",
        timing={
            "discriminator": "duration_timing",
            "start_datetime": "2026-10-06T09:00:00+01:00",
            "duration_minutes": 15,
        },
        recurrence={"discriminator": "recurring", "frequency": "WEEKLY", "count": 4},
    )
    body = service.insert_kwargs["body"]
    assert body["start"]["timeZone"] == "Europe/London"
    assert body["end"]["timeZone"] == "Europe/London"
    assert body["recurrence"] == ["RRULE:FREQ=WEEKLY;COUNT=4"]


def test_create_event_default_start_has_a_time_zone():
    timing = GoogleCalendarCreateEventBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, "event_title": "Planning"}
    ).timing
    assert timing.start_datetime.tzinfo is not None
