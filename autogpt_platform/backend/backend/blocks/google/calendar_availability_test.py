"""Unit tests for the meeting-slot finder and the Suggest Meeting Times block.

2026-10-05 is a Monday. New York is on EDT (UTC-4) until 2026-11-01, then EST.
"""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from backend.blocks.google._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.google._calendar_slots import find_meeting_slots
from backend.blocks.google.calendar_availability import (
    GoogleCalendarSuggestMeetingTimesBlock,
    read_free_busy,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError, BlockInputError

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
HALF_HOUR = timedelta(minutes=30)


def at(day: int, hour: int, minute: int = 0, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


def found(slots) -> list[tuple[datetime, datetime, datetime]]:
    return [(slot.start, slot.end, slot.free_until) for slot in slots]


def monday_9_to_5(busy, **kwargs):
    return find_meeting_slots(busy, at(5, 9), at(5, 17), HALF_HOUR, UTC, **kwargs)


def test_one_slot_at_the_start_of_each_free_period():
    slots = monday_9_to_5([(at(5, 10), at(5, 11)), (at(5, 13), at(5, 14))])
    assert found(slots) == [
        (at(5, 9), at(5, 9, 30), at(5, 10)),
        (at(5, 11), at(5, 11, 30), at(5, 13)),
        (at(5, 14), at(5, 14, 30), at(5, 17)),
    ]


def test_overlapping_nested_and_touching_busy_blocks_merge():
    busy = [
        (at(5, 10), at(5, 11)),
        (at(5, 10, 30), at(5, 12)),
        (at(5, 11, 15), at(5, 11, 45)),
        (at(5, 12), at(5, 12, 30)),
    ]
    assert found(monday_9_to_5(busy)) == [
        (at(5, 9), at(5, 9, 30), at(5, 10)),
        (at(5, 12, 30), at(5, 13), at(5, 17)),
    ]


def test_busy_blocks_from_many_calendars_in_any_order_and_offset():
    busy = [
        (at(5, 13), at(5, 14)),
        (datetime(2026, 10, 5, 6, tzinfo=NEW_YORK), at(5, 11)),
    ]
    assert [slot.start for slot in monday_9_to_5(busy)] == [
        at(5, 9),
        at(5, 11),
        at(5, 14),
    ]


def test_meeting_must_fit_in_the_gap():
    busy = [
        (at(5, 9), at(5, 10)),
        (at(5, 10, 20), at(5, 11)),
        (at(5, 11, 30), at(5, 17)),
    ]
    assert found(monday_9_to_5(busy)) == [(at(5, 11), at(5, 11, 30), at(5, 11, 30))]


def test_window_start_inside_a_busy_block():
    slots = find_meeting_slots(
        [(at(5, 9), at(5, 10))], at(5, 9, 30), at(5, 17), HALF_HOUR, UTC
    )
    assert slots[0].start == at(5, 10)


def test_window_end_cuts_the_last_free_period():
    busy = [(at(5, 9), at(5, 16, 15))]
    cut = find_meeting_slots(busy, at(5, 9), at(5, 16, 40), HALF_HOUR, UTC)
    exact = find_meeting_slots(busy, at(5, 9), at(5, 16, 45), HALF_HOUR, UTC)
    assert cut == []
    assert found(exact) == [(at(5, 16, 15), at(5, 16, 45), at(5, 16, 45))]


def test_busy_time_outside_the_window_is_ignored():
    busy = [(at(4, 9), at(5, 9)), (at(5, 17), at(6, 9))]
    assert found(monday_9_to_5(busy)) == [(at(5, 9), at(5, 9, 30), at(5, 17))]


def test_start_rounds_up_to_the_next_quarter_hour():
    assert monday_9_to_5([(at(5, 9), at(5, 9, 7))])[0].start == at(5, 9, 15)
    assert monday_9_to_5([(at(5, 9), at(5, 9, 15))])[0].start == at(5, 9, 15)


def test_rounding_can_push_a_tight_gap_out():
    busy = [(at(5, 9), at(5, 9, 50)), (at(5, 10, 25), at(5, 17))]
    assert monday_9_to_5(busy) == []


def test_max_slots():
    busy = [(at(5, h, 30), at(5, h + 1)) for h in range(9, 17)]
    assert len(monday_9_to_5(busy, max_slots=20)) == 8
    assert len(monday_9_to_5(busy)) == 5
    assert len(monday_9_to_5(busy, max_slots=3)) == 3


def test_working_hours_apply_to_every_day():
    slots = find_meeting_slots(
        [(at(5, 9), at(5, 17))],
        at(5, 0),
        at(7, 0),
        HALF_HOUR,
        UTC,
        day_start=time(9),
        day_end=time(17),
    )
    assert found(slots) == [(at(6, 9), at(6, 9, 30), at(6, 17))]


def test_meeting_must_be_over_by_the_end_of_the_working_day():
    slots = find_meeting_slots(
        [(at(5, 9), at(5, 16, 45))],
        at(5, 0),
        at(7, 0),
        HALF_HOUR,
        UTC,
        day_start=time(9),
        day_end=time(17),
    )
    assert slots[0].start == at(6, 9)


def test_nothing_outside_working_hours():
    slots = find_meeting_slots(
        [],
        at(5, 17, 30),
        at(6, 8, 59),
        HALF_HOUR,
        UTC,
        day_start=time(9),
        day_end=time(17),
    )
    assert slots == []


def test_only_an_earliest_time_allows_evenings():
    slots = find_meeting_slots(
        [(at(5, 9), at(5, 20))], at(5, 0), at(6, 0), HALF_HOUR, UTC, day_start=time(9)
    )
    assert found(slots) == [(at(5, 20), at(5, 20, 30), at(6, 0))]


@pytest.mark.parametrize(
    "include_weekends, first_start", [(False, at(12, 9)), (True, at(10, 9))]
)
def test_weekends_are_skipped_unless_included(include_weekends, first_start):
    slots = find_meeting_slots(
        [(at(9, 16), at(9, 17))],
        at(9, 16),
        at(12, 12),
        HALF_HOUR,
        UTC,
        day_start=time(9),
        day_end=time(17),
        include_weekends=include_weekends,
    )
    assert slots[0].start == first_start


def test_without_limits_free_time_runs_across_midnight():
    slots = find_meeting_slots([], at(5, 22), at(6, 2), HALF_HOUR, UTC)
    assert found(slots) == [(at(5, 22), at(5, 22, 30), at(6, 2))]


def test_weekdays_only_joins_consecutive_weekdays_and_skips_the_weekend():
    slots = find_meeting_slots(
        [], at(8, 20), at(12, 2), HALF_HOUR, UTC, include_weekends=False
    )
    assert found(slots) == [
        (at(8, 20), at(8, 20, 30), at(10, 0)),
        (at(12, 0), at(12, 0, 30), at(12, 2)),
    ]


def test_working_hours_are_local_to_the_zone():
    slots = find_meeting_slots(
        [], at(5, 0), at(6, 0), HALF_HOUR, NEW_YORK, day_start=time(9), day_end=time(17)
    )
    assert slots[0].start == at(5, 13)
    assert slots[0].start.hour == 9
    assert slots[0].start.utcoffset() == timedelta(hours=-4)
    assert slots[0].free_until == at(5, 21)


def test_working_hours_follow_a_daylight_saving_change():
    slots = find_meeting_slots(
        [],
        at(30, 0),
        at(3, 0, month=11),
        HALF_HOUR,
        NEW_YORK,
        day_start=time(9),
        day_end=time(17),
        include_weekends=False,
    )
    assert [slot.start for slot in slots] == [at(30, 13), at(2, 14, month=11)]
    assert {slot.start.hour for slot in slots} == {9}


def test_weekdays_are_judged_by_the_local_date():
    auckland = ZoneInfo("Pacific/Auckland")
    slots = find_meeting_slots(
        [],
        at(4, 18),
        at(4, 23),
        HALF_HOUR,
        auckland,
        day_start=time(9),
        day_end=time(17),
        include_weekends=False,
    )
    assert found(slots) == [(at(4, 20), at(4, 20, 30), at(4, 23))]
    assert slots[0].start.weekday() == 0


def test_quarter_hours_are_local_in_a_45_minute_offset_zone():
    kathmandu = ZoneInfo("Asia/Kathmandu")
    slots = find_meeting_slots(
        [(at(5, 3, 15), at(5, 3, 20))],
        at(5, 3),
        at(5, 6),
        HALF_HOUR,
        kathmandu,
        day_start=time(9),
    )
    assert slots[0].start.astimezone(kathmandu).time() == time(9, 15)
    assert slots[0].start == at(5, 3, 30)


def test_no_slots_when_the_meeting_is_longer_than_any_gap():
    assert monday_9_to_5([(at(5, 12), at(5, 13))], max_slots=5) != []
    slots = find_meeting_slots(
        [(at(5, 12), at(5, 13))], at(5, 9), at(5, 17), timedelta(hours=5), UTC
    )
    assert slots == []


def _suggest_input(**fields) -> GoogleCalendarSuggestMeetingTimesBlock.Input:
    return GoogleCalendarSuggestMeetingTimesBlock.Input.model_validate(
        {"credentials": TEST_CREDENTIALS_INPUT, **fields}
    )


def test_calendar_ids_include_me_and_dedupe_attendees():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    ids = block._calendar_ids(
        _suggest_input(attendee_emails=["a@x.com", " A@x.com ", "", "b@x.com"])
    )
    assert ids == ["primary", "a@x.com", "b@x.com"]


def test_calendar_ids_need_someone_and_at_most_fifty():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    with pytest.raises(BlockInputError, match="at least one"):
        block._calendar_ids(_suggest_input(include_me=False))
    many = [f"p{i}@x.com" for i in range(50)]
    with pytest.raises(BlockInputError, match="at most 50"):
        block._calendar_ids(_suggest_input(attendee_emails=many))


def test_working_hours_parsing():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    assert block._working_hours(_suggest_input()) == (time(9), time(17))
    assert block._working_hours(
        _suggest_input(earliest_time="8:30", latest_time="18:00:00")
    ) == (time(8, 30), time(18))
    assert block._working_hours(_suggest_input(earliest_time="", latest_time="")) == (
        None,
        None,
    )
    for bad in [{"earliest_time": "9am"}, {"latest_time": "24:00"}]:
        with pytest.raises(BlockInputError, match="time of day"):
            block._working_hours(_suggest_input(**bad))
    with pytest.raises(BlockInputError, match="later than"):
        block._working_hours(_suggest_input(earliest_time="17:00", latest_time="9:00"))


def test_window_defaults_to_the_next_week_and_never_starts_in_the_past():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    now = at(5, 10, 7)
    assert block._window(_suggest_input(), UTC, now) == (now, now + timedelta(days=7))
    start, end = block._window(
        _suggest_input(
            window_start="2026-10-01T09:00:00Z", window_end="2026-10-06T00:00:00Z"
        ),
        UTC,
        now,
    )
    assert (start, end) == (now, at(6, 0))


def test_window_times_without_offset_use_the_zone():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    start, end = block._window(
        _suggest_input(window_start="2026-10-06T09:00", window_end="2026-10-06T17:00"),
        NEW_YORK,
        at(5, 0),
    )
    assert (start, end) == (at(6, 13), at(6, 21))


def test_window_in_the_past_is_rejected():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    with pytest.raises(BlockInputError, match="in the future"):
        block._window(_suggest_input(window_end="2026-10-01T00:00:00Z"), UTC, at(5, 0))


def test_read_free_busy_splits_readable_and_refused_calendars():
    free_busy = read_free_busy(
        {
            "groups": {"team@x.com": {"errors": [{"reason": "groupTooBig"}]}},
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": "2026-10-05T09:00:00Z", "end": "2026-10-05T10:00:00Z"}
                    ]
                },
                "a@x.com": {"busy": []},
                "b@partner.com": {"errors": [{"reason": "notFound"}]},
            },
        }
    )
    assert free_busy.busy == [(at(5, 9), at(5, 10))]
    assert free_busy.checked == ["primary", "a@x.com"]
    assert free_busy.unchecked == ["team@x.com", "b@partner.com"]


class _FreeBusyService:
    def __init__(self):
        self.body: dict = {}

    def freebusy(self):
        return self

    def query(self, body):
        self.body = body
        return self

    def execute(self):
        return {}


def test_free_busy_query_asks_in_utc():
    service = _FreeBusyService()
    GoogleCalendarSuggestMeetingTimesBlock._query_free_busy(
        service,
        ["primary", "a@x.com"],
        datetime(2026, 10, 5, 9, 0, 12, 345, tzinfo=NEW_YORK),
        at(6, 0),
    )
    assert service.body == {
        "timeMin": "2026-10-05T13:00:12+00:00",
        "timeMax": "2026-10-06T00:00:00+00:00",
        "items": [{"id": "primary"}, {"id": "a@x.com"}],
    }


async def _run_suggest(block, context: ExecutionContext, **fields):
    return [
        output
        async for output in block.run(
            _suggest_input(**fields),
            credentials=TEST_CREDENTIALS,
            execution_context=context,
        )
    ]


@pytest.mark.asyncio
async def test_suggest_uses_the_profile_time_zone_by_default():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    block._query_free_busy = lambda *args: {"calendars": {"primary": {"busy": []}}}
    outputs = await _run_suggest(
        block,
        ExecutionContext(user_timezone="America/New_York"),
        window_start="2099-01-05T00:00:00Z",
        window_end="2099-01-06T00:00:00Z",
    )
    first = dict(outputs)["slots"][0]
    assert first.start == datetime(2099, 1, 5, 14, tzinfo=UTC)
    assert ("time_zone", "America/New_York") in outputs


@pytest.mark.asyncio
async def test_suggest_fails_when_no_calendar_is_readable():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    block._query_free_busy = lambda *args: {
        "calendars": {"a@partner.com": {"errors": [{"reason": "notFound"}]}}
    }
    with pytest.raises(BlockExecutionError, match="a@partner.com"):
        await _run_suggest(
            block,
            ExecutionContext(),
            attendee_emails=["a@partner.com"],
            include_me=False,
        )


@pytest.mark.asyncio
async def test_suggest_rejects_an_unknown_time_zone():
    block = GoogleCalendarSuggestMeetingTimesBlock()
    with pytest.raises(BlockInputError, match="time zone"):
        await _run_suggest(block, ExecutionContext(), time_zone="Mars/Olympus_Mons")
