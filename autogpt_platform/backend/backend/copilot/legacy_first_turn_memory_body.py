"""What the renderers wrote inside the first-turn memory block older sessions
hold, between its ``<temporal_context>`` tags: ``is_rendered`` says whether one
of them could have written a given body.

Production's renderer (``graphiti/context._format_context`` with the helpers
of ``graphiti/_format.py``, unchanged on master from #12720 to this change)
wrote ``  - {fact} ({valid_from} — {valid_to})`` for a fact and
``  - [{created_at}] {body[:500]}`` for an episode, each time ``str()`` of a
datetime (``unknown`` and ``present`` when unset), the ``<FACTS>`` and
``<RECENT_EPISODES>`` sections in that order with at least one line each.
This stack's recall policy (``graphiti/recall_render.py``) writes
``(valid: {from} — {to})``, or how and when a fact was retired, instead; since
it neutralised tag starts in memory text (``<`` became ``<!``, after the cut),
an episode's body can run past 500 characters by one a tag.

A body is theirs when every part of it is what one renderer writes:

- every fact line ends in a stamp of that renderer, and every episode line
  opens on a bracketed ``created_at``;
- every time is ``str()`` of a datetime that exists, in ASCII digits, written
  exactly as ``str()`` writes it;
- every episode's body is no longer than the cut left it: 500 characters, or
  500 without the marks this stack's renderer added, when no line holds a tag
  start it would have neutralised.

A line whose text holds a newline followed by ``  - `` reads as two items,
each of which must pass on its own; a block where one does not is left, a
miss on the safe side.
"""

import re
from datetime import datetime, timedelta, timezone

_FACTS = ("<FACTS>\n", "\n</FACTS>")
_EPISODES = ("<RECENT_EPISODES>\n", "\n</RECENT_EPISODES>")
_BETWEEN_SECTIONS = f"{_FACTS[1]}\n\n{_EPISODES[0]}"
_ITEM = "  - "
# The shape of ``str()`` of a datetime, in ASCII digits: microseconds when
# set, the offset when aware. ``_is_str_of_datetime`` checks the value, from
# the parts it captures.
_DATETIME = (
    r"([0-9]{4})-([0-9]{2})-([0-9]{2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{6}))?"
    r"(?:([+-])([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{6}))?)?)?"
)
_DATETIME_RE = re.compile(_DATETIME)
# What closes a fact line: production's ``({from} — {to})``, or this stack's
# ``(valid: {from} — {to})`` and how and when a fact was retired.
_FACT_STAMPS = {
    "production": re.compile(
        rf"(?P<start>{_DATETIME}|unknown) — (?P<end>{_DATETIME}|present)"
    ),
    "stack": re.compile(
        rf"valid: (?P<start>{_DATETIME}|unknown) — (?P<end>{_DATETIME}|present)"
        rf"|(?:superseded|contradicted|retracted|expired)"
        rf" (?P<retired>{_DATETIME}|at an unknown time)"
    ),
}
_UNSET_TIMES = ("unknown", "present", "at an unknown time")
_EPISODE_STAMP_RE = re.compile(rf"\[(?P<created>{_DATETIME})\] ")
# The renderers cut an episode's body to this many characters.
_EPISODE_BODY_CHARS = 500
# A tag start this stack's renderer would have neutralised, and one it did
# (``recall_render.neutralise_tags``: ``<`` became ``<!``). The first is the
# renderer's own linear-time pattern (#15003 replaced its earlier
# ``\s*/?\s*``, which backtracked quadratically over a whitespace run).
_TAG_START_RE = re.compile(r"<(?=\s*(?:/\s*)?[^\W\d])")
_NEUTRALISED_TAG_START_RE = re.compile(r"<!(?=\s*(?:/\s*)?[^\W\d])")


def is_rendered(body: str) -> bool:
    """Whether a renderer could have written ``body``: ``<FACTS>``,
    ``<RECENT_EPISODES>`` or both, in that order (see the module docstring).
    It reads the body once, without backtracking."""
    if body.startswith(_EPISODES[0]):
        return _is_written([], _items(body, _EPISODES))
    if body.endswith(_FACTS[1]) and _is_written(_items(body, _FACTS), []):
        return True
    between = body.find(_BETWEEN_SECTIONS)
    if between == -1:
        return False
    facts_end = between + len(_FACTS[1])
    episodes_start = between + len(_BETWEEN_SECTIONS) - len(_EPISODES[0])
    return _is_written(
        _items(body[:facts_end], _FACTS), _items(body[episodes_start:], _EPISODES)
    )


def _items(section: str, tags: tuple[str, str]) -> list[str] | None:
    """The ``  - `` items between a section's tags, without the marker; None
    when the section is not wrapped in them or does not open on an item."""
    opening, closing = tags
    if len(section) < len(opening) + len(closing):
        return None
    if not (section.startswith(opening) and section.endswith(closing)):
        return None
    lines = section[len(opening) : -len(closing)]
    if not lines.startswith(_ITEM):
        return None
    return lines[len(_ITEM) :].split(f"\n{_ITEM}")


def _is_written(facts: list[str] | None, episodes: list[str] | None) -> bool:
    """Whether one renderer wrote these fact and episode lines, None for a
    section that is not one.

    Every fact ends with the stamp of the same renderer, production's or this
    stack's. Every episode opens on its ``created_at``, and its body is no
    longer than the cut left it: 500 characters, or, as this stack's renderer
    wrote it, 500 once the marks it added are taken out, when no line in the
    block holds a tag start it would have neutralised.
    """
    if facts is None or episodes is None:
        return False
    renderers = {_fact_renderer(fact) for fact in facts}
    bodies = _episode_bodies(episodes)
    if bodies is None or None in renderers or len(renderers) > 1:
        return False
    if all(len(body) <= _EPISODE_BODY_CHARS for body in bodies):
        return True
    return (
        renderers <= {"stack"}
        and not any(_TAG_START_RE.search(line) for line in facts + episodes)
        and all(
            len(body) - len(_NEUTRALISED_TAG_START_RE.findall(body))
            <= _EPISODE_BODY_CHARS
            for body in bodies
        )
    )


def _fact_renderer(item: str) -> str | None:
    """Which renderer's stamp closes ``item`` as `` (stamp)``, or None when
    neither's does or a time in it is not ``str()`` of a real datetime."""
    opening = item.rfind(" (")
    if opening == -1 or not item.endswith(")"):
        return None
    for renderer, stamp_re in _FACT_STAMPS.items():
        stamp = stamp_re.fullmatch(item, opening + 2, len(item) - 1)
        if stamp is not None:
            times = stamp.groupdict().values()
            real = all(
                time is None or time in _UNSET_TIMES or _is_str_of_datetime(time)
                for time in times
            )
            return renderer if real else None
    return None


def _episode_bodies(episodes: list[str]) -> list[str] | None:
    """Each episode's body after its ``[created_at] ``, or None when one does
    not open on ``str()`` of a real datetime."""
    bodies: list[str] = []
    for episode in episodes:
        stamp = _EPISODE_STAMP_RE.match(episode)
        if stamp is None or not _is_str_of_datetime(stamp["created"]):
            return None
        bodies.append(episode[stamp.end() :])
    return bodies


def _is_str_of_datetime(text: str) -> bool:
    """Whether ``text`` is ``str()`` of a datetime: a date, time and offset
    that exist, written exactly as ``str()`` writes them (microseconds and
    offset seconds only when set, a zero offset as ``+00:00``).

    The datetime is built from the parts rather than read back with
    ``datetime.fromisoformat``, which drops an offset's microseconds when its
    hours, minutes and seconds are all zero.
    """
    parts = _DATETIME_RE.fullmatch(text)
    if parts is None:
        return False
    date_time, offset = parts.groups()[:7], parts.groups()[7:]
    sign, *offset_parts = offset
    try:
        tz = None
        if sign is not None:
            hours, minutes, seconds, micros = (int(n or 0) for n in offset_parts)
            span = timedelta(
                hours=hours, minutes=minutes, seconds=seconds, microseconds=micros
            )
            tz = timezone(-span if sign == "-" else span)
        value = datetime(*(int(n or 0) for n in date_time), tzinfo=tz)
    except ValueError:
        return False
    return str(value) == text
