import re
from calendar import day_abbr, day_name, month_abbr, month_name
from datetime import datetime as datetime_
from datetime import timedelta, timezone
from time import localtime, strftime

tokens = r"H{1,2}|h{1,2}|m{1,2}|s{1,2}|S{1,6}|YYYY|YY|M{1,4}|D{1,4}|Z{1,2}|zz|A|X|x|E|Q|dddd|ddd|d"

pattern = re.compile(r"(?:{0})|\[(?:{0}|!UTC)\]".format(tokens))


class datetime(datetime_):  # noqa: N801
    def __format__(self, spec):
        if spec.endswith("!UTC"):
            dt = self.astimezone(timezone.utc)
            spec = spec[:-4]
        else:
            dt = self

        if not spec:
            spec = "%Y-%m-%dT%H:%M:%S.%f%z"

        if "%" in spec:
            return datetime_.__format__(dt, spec)

        year, month, day, hour, minute, second, weekday, yearday, _ = dt.timetuple()
        microsecond = dt.microsecond
        timestamp = dt.timestamp()
        tzinfo = dt.tzinfo or timezone(timedelta(seconds=0))
        offset = tzinfo.utcoffset(dt).total_seconds()
        sign = ("-", "+")[offset >= 0]
        h, m = divmod(abs(offset // 60), 60)

        rep = {
            "YYYY": "%04d" % year,
            "YY": "%02d" % (year % 100),
            "Q": "%d" % ((month - 1) // 3 + 1),
            "MMMM": month_name[month],
            "MMM": month_abbr[month],
            "MM": "%02d" % month,
            "M": "%d" % month,
            "DDDD": "%03d" % yearday,
            "DDD": "%d" % yearday,
            "DD": "%02d" % day,
            "D": "%d" % day,
            "dddd": day_name[weekday],
            "ddd": day_abbr[weekday],
            "d": "%d" % weekday,
            "E": "%d" % (weekday + 1),
            "HH": "%02d" % hour,
            "H": "%d" % hour,
            "hh": "%02d" % ((hour - 1) % 12 + 1),
            "h": "%d" % ((hour - 1) % 12 + 1),
            "mm": "%02d" % minute,
            "m": "%d" % minute,
            "ss": "%02d" % second,
            "s": "%d" % second,
            "S": "%d" % (microsecond // 100000),
            "SS": "%02d" % (microsecond // 10000),
            "SSS": "%03d" % (microsecond // 1000),
            "SSSS": "%04d" % (microsecond // 100),
            "SSSSS": "%05d" % (microsecond // 10),
            "SSSSSS": "%06d" % microsecond,
            "A": ("AM", "PM")[hour // 12],
            "Z": "%s%02d:%02d" % (sign, h, m),
            "ZZ": "%s%02d%02d" % (sign, h, m),
            "zz": tzinfo.tzname(dt) or "",
            "X": "%d" % timestamp,
            "x": "%d" % (int(timestamp) * 1000000 + microsecond),
        }

        def get(m):
            try:
                return rep[m.group(0)]
            except KeyError:
                return m.group(0)[1:-1]

        return pattern.sub(get, spec)


def aware_now():
    now = datetime_.now()
    timestamp = now.timestamp()
    local = localtime(timestamp)

    try:
        seconds = local.tm_gmtoff
        zone = local.tm_zone
    except AttributeError:
        offset = datetime_.fromtimestamp(timestamp) - datetime_.utcfromtimestamp(timestamp)
        seconds = offset.total_seconds()
        zone = strftime("%Z")

    tzinfo = timezone(timedelta(seconds=seconds), zone)

    return datetime.combine(now.date(), now.time().replace(tzinfo=tzinfo))
