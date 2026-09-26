import asyncio
import re
from datetime import datetime, time, timedelta, timezone, tzinfo
from typing import Any

from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

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
    CALENDAR_READONLY_SCOPE,
    build_calendar_service,
    calendar_error,
    in_zone,
    resolve_time_zone,
)
from ._calendar_slots import Interval, MeetingSlot, find_meeting_slots

DEFAULT_WINDOW = timedelta(days=7)
MAX_CALENDARS = 50

_TIME_OF_DAY = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*$")


def _test_monday() -> datetime:
    """Midnight UTC on a Monday at least a week away, so the self-test never
    asks for times in the past."""
    today = datetime.now(timezone.utc).date()
    monday = today + timedelta(days=14 - today.weekday())
    return datetime.combine(monday, time.min, tzinfo=timezone.utc)


class GoogleCalendarSuggestMeetingTimesBlock(Block):
    """Suggest times when everyone is free, from Google Calendar free/busy."""

    class Input(BlockSchemaInput):
        credentials: GoogleCredentialsInput = GoogleCredentialsField(
            [CALENDAR_READONLY_SCOPE]
        )
        attendee_emails: list[str] = SchemaField(
            description="Email addresses of the people who need to be free. Google only shares free/busy for calendars you can see, usually people in your organization.",
            default_factory=list,
        )
        include_me: bool = SchemaField(
            description="Also check your own main calendar", default=True
        )
        duration_minutes: int = SchemaField(
            description="Meeting length in minutes", default=30, ge=5, le=1440
        )
        window_start: datetime | None = SchemaField(
            description="Earliest time to suggest. Empty means now.", default=None
        )
        window_end: datetime | None = SchemaField(
            description="Time by which the meeting must be over. Empty means 7 days after the start.",
            default=None,
        )
        time_zone: str = SchemaField(
            description="Time zone for the working hours, for window times without a UTC offset and for the suggestions, e.g. Europe/London. Empty uses your profile time zone.",
            default="",
        )
        earliest_time: str = SchemaField(
            description="Don't suggest meetings that start before this time of day (HH:MM, 24-hour). Empty allows any time.",
            default="09:00",
        )
        latest_time: str = SchemaField(
            description="Meetings must be over by this time of day (HH:MM, 24-hour). Empty allows any time.",
            default="17:00",
        )
        include_weekends: bool = SchemaField(
            description="Also suggest times on Saturdays and Sundays", default=False
        )
        max_suggestions: int = SchemaField(
            description="Maximum number of times to suggest", default=5, ge=1, le=50
        )

    class Output(BlockSchemaOutput):
        slots: list[MeetingSlot] = SchemaField(
            description="Suggested meeting times, earliest first, at most one per free period"
        )
        slot: MeetingSlot = SchemaField(description="Each suggested meeting time")
        unchecked_calendars: list[str] = SchemaField(
            description="Calendars whose free/busy Google wouldn't share, e.g. people outside your organization. The suggestions ignore them."
        )
        time_zone: str = SchemaField(
            description="Time zone of the suggestions and the working hours"
        )

    def __init__(self):
        monday = _test_monday()

        def at(hour: int, minute: int = 0) -> datetime:
            return monday + timedelta(hours=hour, minutes=minute)

        test_slots = [
            MeetingSlot(start=at(10), end=at(10, 30), free_until=at(10, 30)),
            MeetingSlot(start=at(12), end=at(12, 30), free_until=at(17)),
        ]
        super().__init__(
            id="2f354276-8aef-4813-a850-3bcb88adbfaf",
            description=(
                "Suggest meeting times when you and the given people are all free, "
                "using Google Calendar free/busy. By default only suggests weekday "
                "times between 09:00 and 17:00 in your time zone."
            ),
            categories={BlockCategory.PRODUCTIVITY},
            input_schema=GoogleCalendarSuggestMeetingTimesBlock.Input,
            output_schema=GoogleCalendarSuggestMeetingTimesBlock.Output,
            disabled=not GOOGLE_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "attendee_emails": ["alex@example.com", "sam@partner.example"],
                "window_start": monday.isoformat(),
                "window_end": at(24).isoformat(),
                "time_zone": "UTC",
                "max_suggestions": 2,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("slots", test_slots),
                ("slot", test_slots[0]),
                ("slot", test_slots[1]),
                ("unchecked_calendars", ["sam@partner.example"]),
                ("time_zone", "UTC"),
            ],
            test_mock={
                "_query_free_busy": lambda *args, **kwargs: {
                    "calendars": {
                        "primary": {
                            "busy": [
                                {"start": at(9).isoformat(), "end": at(10).isoformat()}
                            ]
                        },
                        "alex@example.com": {
                            "busy": [
                                {
                                    "start": at(10, 30).isoformat(),
                                    "end": at(12).isoformat(),
                                }
                            ]
                        },
                        "sam@partner.example": {
                            "errors": [{"domain": "global", "reason": "notFound"}]
                        },
                    }
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
        zone = resolve_time_zone(
            input_data.time_zone, execution_context.user_timezone, self.name, self.id
        )
        day_start, day_end = self._working_hours(input_data)
        start, end = self._window(input_data, zone, datetime.now(timezone.utc))
        calendar_ids = self._calendar_ids(input_data)
        service = build_calendar_service(credentials)
        try:
            result = await asyncio.to_thread(
                self._query_free_busy, service, calendar_ids, start, end
            )
        except HttpError as e:
            raise calendar_error(e, self.name, self.id) from e

        free_busy = read_free_busy(result)
        if not free_busy.checked:
            raise BlockExecutionError(
                message=(
                    "Google wouldn't share free/busy for any of these calendars: "
                    f"{', '.join(free_busy.unchecked or calendar_ids)}."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        slots = find_meeting_slots(
            free_busy.busy,
            start,
            end,
            timedelta(minutes=input_data.duration_minutes),
            zone,
            day_start=day_start,
            day_end=day_end,
            include_weekends=input_data.include_weekends,
            max_slots=input_data.max_suggestions,
        )
        yield "slots", slots
        for slot in slots:
            yield "slot", slot
        yield "unchecked_calendars", free_busy.unchecked
        yield "time_zone", zone.key

    def _working_hours(self, input_data: Input) -> tuple[time | None, time | None]:
        day_start = self._time_of_day(input_data.earliest_time, "earliest_time")
        day_end = self._time_of_day(input_data.latest_time, "latest_time")
        if day_end is not None and day_end <= (day_start or time.min):
            raise self._input_error("latest_time must be later than earliest_time.")
        return day_start, day_end

    def _time_of_day(self, value: str, field: str) -> time | None:
        if not value.strip():
            return None
        match = _TIME_OF_DAY.match(value)
        if match and int(match[1]) < 24 and int(match[2]) < 60:
            return time(int(match[1]), int(match[2]))
        raise self._input_error(f"{field} should be a time of day like 09:00 or 17:30.")

    def _window(
        self, input_data: Input, zone: tzinfo, now: datetime
    ) -> tuple[datetime, datetime]:
        """The search window in UTC, never starting in the past."""
        start = now
        if input_data.window_start:
            start = max(
                in_zone(input_data.window_start, zone).astimezone(timezone.utc), now
            )
        end = (
            in_zone(input_data.window_end, zone).astimezone(timezone.utc)
            if input_data.window_end
            else start + DEFAULT_WINDOW
        )
        if end <= start:
            raise self._input_error(
                "The time window must end after it starts, and in the future."
            )
        return start, end

    def _calendar_ids(self, input_data: Input) -> list[str]:
        ids = ["primary"] if input_data.include_me else []
        seen: set[str] = set()
        for email in (email.strip() for email in input_data.attendee_emails):
            if email and email.lower() not in seen:
                ids.append(email)
                seen.add(email.lower())
        if not ids:
            raise self._input_error(
                "Add at least one attendee email, or turn on include_me."
            )
        if len(ids) > MAX_CALENDARS:
            raise self._input_error(
                f"Google checks at most {MAX_CALENDARS} calendars at a time."
            )
        return ids

    def _input_error(self, message: str) -> BlockInputError:
        return BlockInputError(message=message, block_name=self.name, block_id=self.id)

    @staticmethod
    def _query_free_busy(
        service, calendar_ids: list[str], start: datetime, end: datetime
    ) -> dict:
        body = {
            "timeMin": start.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "timeMax": end.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "items": [{"id": calendar_id} for calendar_id in calendar_ids],
        }
        return service.freebusy().query(body=body).execute()


class FreeBusy(BaseModel):
    busy: list[Interval] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    unchecked: list[str] = Field(default_factory=list)


def read_free_busy(result: dict[str, Any]) -> FreeBusy:
    """Busy intervals from the calendars Google shared, and the ones it refused."""
    free_busy = FreeBusy(
        unchecked=[
            group
            for group, info in result.get("groups", {}).items()
            if info.get("errors")
        ]
    )
    for calendar_id, info in result.get("calendars", {}).items():
        if info.get("errors"):
            free_busy.unchecked.append(calendar_id)
            continue
        free_busy.checked.append(calendar_id)
        free_busy.busy.extend(
            (datetime.fromisoformat(b["start"]), datetime.fromisoformat(b["end"]))
            for b in info.get("busy", [])
        )
    return free_busy
