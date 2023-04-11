from datetime import date, datetime, time, timedelta, timezone, tzinfo
import re
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from re import Match

    from pip._vendor.tomli._parser import ParseFloat

# E.g.
# - 00:32:00.999999
# - 00:32:00
_TIME_RE_STR = r"([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?"

RE_HEX = re.compile(r"[0-9A-Fa-f](?:_?[0-9A-Fa-f])*")
RE_BIN = re.compile(r"[01](?:_?[01])*")
RE_OCT = re.compile(r"[0-7](?:_?[0-7])*")
RE_NUMBER = re.compile(
    r"[+-]?(?:0|[1-9](?:_?[0-9])*)"  # integer
    + r"(?:\.[0-9](?:_?[0-9])*)?"  # optional fractional part
    + r"(?:[eE][+-]?[0-9](?:_?[0-9])*)?"  # optional exponent part
)
RE_LOCALTIME = re.compile(_TIME_RE_STR)
RE_DATETIME = re.compile(
    r"([0-9]{4})-(0[1-9]|1[0-2])-(0[1-9]|1[0-9]|2[0-9]|3[01])"  # date, e.g. 1988-10-27
    + r"(?:"
    + r"[T ]"
    + _TIME_RE_STR
    + r"(?:(Z)|([+-])([01][0-9]|2[0-3]):([0-5][0-9]))?"  # time offset
    + r")?"
)


def match_to_datetime(match: "Match") -> Union[datetime, date]:
    """Convert a `RE_DATETIME` match to `datetime.datetime` or `datetime.date`.

    Raises ValueError if the match does not correspond to a valid date
    or datetime.
    """
    (
        year_str,
        month_str,
        day_str,
        hour_str,
        minute_str,
        sec_str,
        micros_str,
        zulu_time,
        offset_dir_str,
        offset_hour_str,
        offset_minute_str,
    ) = match.groups()
    year, month, day = int(year_str), int(month_str), int(day_str)
    if hour_str is None:
        return date(year, month, day)
    hour, minute, sec = int(hour_str), int(minute_str), int(sec_str)
    micros = int(micros_str[1:].ljust(6, "0")[:6]) if micros_str else 0
    if offset_dir_str:
        offset_dir = 1 if offset_dir_str == "+" else -1
        tz: Optional[tzinfo] = timezone(
            timedelta(
                hours=offset_dir * int(offset_hour_str),
                minutes=offset_dir * int(offset_minute_str),
            )
        )
    elif zulu_time:
        tz = timezone.utc
    else:  # local date-time
        tz = None
    return datetime(year, month, day, hour, minute, sec, micros, tzinfo=tz)


def match_to_localtime(match: "Match") -> time:
    hour_str, minute_str, sec_str, micros_str = match.groups()
    micros = int(micros_str[1:].ljust(6, "0")[:6]) if micros_str else 0
    return time(int(hour_str), int(minute_str), int(sec_str), micros)


def match_to_number(match: "Match", parse_float: "ParseFloat") -> Any:
    match_str = match.group()
    if "." in match_str or "e" in match_str or "E" in match_str:
        return parse_float(match_str)
    return int(match_str)
