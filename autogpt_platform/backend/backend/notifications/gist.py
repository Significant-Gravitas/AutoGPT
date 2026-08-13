"""Turning a run's outputs into one sentence for the Briefing's highlights.

Strict rules, because this is what replaces dumping raw output into the inbox:

* a hard character limit
* text outputs get one sentence
* structured outputs get counted and characterised
* binary and file outputs get named, never embedded
* on failure, fall back to a truncated first line, then to "Completed N runs."

Never inline a raw output. The inbox gets the gist and the link; the platform
is where outputs live.
"""

import re
from typing import Any

GIST_MAX_CHARS = 150

# Output names that read as files rather than prose, so they are named by kind
# and count instead of being quoted.
_FILE_HINT = re.compile(r"(file|image|video|audio|clip|pdf|doc|csv|sheet)", re.I)
_DATA_URI = re.compile(r"^data:([\w.+-]+)/")
_URL = re.compile(r"^https?://")


def build_gist(
    outputs: dict[str, list[Any]], activity_status: str | None
) -> str | None:
    """One sentence describing what a run produced, or None when there is
    nothing honest to say — the caller then falls back to counts and links."""
    if activity_status and activity_status.strip():
        return _one_sentence(activity_status)

    parts = [
        p for p in (_describe(name, values) for name, values in outputs.items()) if p
    ]
    if not parts:
        return None
    return _truncate(_join(parts))


def fallback_gist(run_count: int) -> str:
    """The last fallback. Says only what we actually know."""
    runs = "1 run" if run_count == 1 else f"{run_count} runs"
    return f"completed {runs}."


def _describe(name: str, values: list[Any]) -> str | None:
    if not values:
        return None
    if _looks_like_files(name, values):
        return _describe_files(name, values)
    if len(values) == 1 and isinstance(values[0], (dict, list)):
        return _describe_structured(name, values[0])
    if len(values) > 1:
        return f"produced {len(values)} {_plural(name)}"
    return _describe_text(values[0])


def _describe_files(name: str, values: list[Any]) -> str:
    """Named, never embedded."""
    count = len(values)
    kind = _plural(name) if count != 1 else name
    return f"produced {count} {kind}"


def _describe_structured(name: str, value: dict | list) -> str:
    """Counted and characterised, not pasted."""
    if isinstance(value, list):
        return f"produced {len(value)} {_plural(name)}"
    keys = len(value)
    field_word = "field" if keys == 1 else "fields"
    return f"produced a {name} with {keys} {field_word}"


def _describe_text(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    words = len(text.split())
    if words > 60:
        return f"wrote a {words:,}-word result"
    return _one_sentence(text)


def _one_sentence(text: str) -> str:
    """First sentence only, collapsed onto one line and length-capped."""
    flat = " ".join(text.split())
    match = re.search(r"(?<=[.!?])\s", flat)
    sentence = flat[: match.start()] if match else flat
    return _truncate(sentence)


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return _finish(parts[0])
    return _finish(", ".join(parts[:-1]) + f" and {parts[-1]}")


def _finish(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def _truncate(text: str) -> str:
    if len(text) <= GIST_MAX_CHARS:
        return _finish(text)
    cut = text[: GIST_MAX_CHARS - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-") + "…"


def _plural(name: str) -> str:
    word = name.replace("_", " ").strip() or "result"
    if word.endswith("s"):
        return word
    return word + "s"


def _looks_like_files(name: str, values: list[Any]) -> bool:
    if _FILE_HINT.search(name):
        return True
    return any(
        isinstance(v, str) and (_DATA_URI.match(v) or _URL.match(v)) for v in values
    )
