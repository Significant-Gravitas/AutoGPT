"""Secret-key redaction for shared chats.

A shared chat shows its tool calls and tool results to anyone holding the
link, so every value under a secret-shaped key is replaced with
:data:`_REDACTED` before they leave the backend: in dicts and lists, and in
JSON carried inside a string, which is how both engines store a tool call's
arguments.  An approved call's late result is tool output too, though it is
stored in a user row.

What can't be checked is withheld instead of shown: arguments that don't
parse; a tool result that starts like JSON but doesn't parse; the preview
inside a ``<tool-output-truncated>`` result; JSON cut in the middle, which is
how a result is stored when it is too large and its full copy can't be
saved; JSON too large or too deeply nested to read; and any other text that
doesn't parse but quotes a secret-shaped key before a colon, such as a
window or preview of a JSON file, JSON lines, or a dict's repr.  Other text
is never key-redacted, and is shown as written, as is JSON with nothing in it
to redact.

What a read returns from one of Otto's working files, such as a large tool
output saved in full, is withheld whole, as that output's stored preview is:
a window of it, a preview of it, or the whole file in base64.

Called from :func:`backend.copilot.sharing.models.sanitize_chat_message`.
"""

import json
import re
from typing import Any

from backend.data.workspace_internal import is_internal_workspace_file

# Secret-shaped key hints.  Each entry is a whole "word" — boundary
# matching is enforced by :data:`_SECRET_KEY_RE` so that ``token`` does
# not false-positive against ``prompt_tokens`` (LLM usage metadata) and
# ``auth`` does not false-positive against ``author`` (attribution).
_SECRET_KEY_HINTS = (
    "api_key",
    "api_keys",
    "api_token",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "oauth",
    "passwd",
    "password",
    "secret",
    "secrets",
    "token",
)
_REDACTED = "[redacted]"

# Boundary-aware matcher.  ``(?<![a-z0-9])`` and ``(?![a-z0-9])`` reject
# matches where a letter/digit sits on either side of the hint, so:
#
#   - ``access_token``  matches ``token``   (``_`` boundary, end)
#   - ``total_tokens``  does NOT match      (``s`` letter follows)
#   - ``author``        does NOT match      (``o`` letter follows ``auth``)
#   - ``Authorization`` matches ``authorization`` whole, case-insensitive
#
# Using these lookarounds rather than ``\b`` is deliberate: ``\b`` treats
# underscore as a word character, which would mis-classify the ``_`` →
# letter transition in ``total_tokens`` as a non-boundary and leak.
_SECRET_KEY_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(h) for h in _SECRET_KEY_HINTS)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)
_CAMEL_CASE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")

# Real calls and results nest a few levels.  Below this the walk withholds
# what it finds, so a deeply nested payload can't exhaust the stack.
_MAX_DEPTH = 100

_JSON_CONTAINER_RE = re.compile(r"\s*[\[{]")

# What ``copilot/tools/base.py`` stores for a result too large to keep inline.
# Its preview, a cut or an outline of the JSON, can't be redacted by key: a
# cut can begin inside a value whose key was cut away, and the outline doesn't
# quote its keys or values.
_TRUNCATED_RESULT_RE = re.compile(
    r"\s*<tool-output-truncated\b(?:\s+total_chars=(\d+))?"
)

# What ``backend.util.truncate`` puts where it cut the middle out of a text.
# JSON cut that way can still parse, with the end of one value (a secret whose
# key was cut away) joined onto the value the kept start ended in.
_CUT_MARKER_RE = re.compile(r"… \(omitted \d+ chars\)…")

# A quoted key and the colon after it.  In JSON quoted inside a string the key
# ends in the backslash before its closing quote, which :func:`_is_secret_key`
# reads as a word boundary.
_QUOTED_KEY_RE = re.compile(r"""["']([^"'\n]{1,200})["']\s*:""")

# How ``copilot/gate/held.py`` stores an approved call's late result: the
# tool's output in a user row, after an opening tag on a line of its own whose
# attribute values are quoted.
_HELD_RESULT_TAG = "<held_call_result"
_HELD_RESULT_RE = re.compile(
    r'(<held_call_result(?: [a-z_]+="[^"\n]*")*>\n)(.*)(\n</held_call_result>)',
    re.DOTALL,
)
_HELD_RESULT_WITHHELD = f"{_HELD_RESULT_TAG}>\n{_REDACTED}\n</held_call_result>"

# What a read_workspace_file result returns from the file it names: a window
# or the whole text, the whole file in base64, or a preview of its head.
_READ_BACK_KEYS = frozenset({"content", "content_base64", "preview"})


def redact_tool_calls(tool_calls: list[dict] | None) -> list[dict] | None:
    if tool_calls is None:
        return None
    return [redact_tool_call(call) for call in tool_calls]


def redact_tool_call(call: Any) -> Any:
    """Redact one tool call, or a message's legacy ``function_call``.

    Its arguments, at the top or under ``function``, are JSON by contract, so
    a string of them that doesn't parse (cut off mid-stream, say) can't be
    checked and is withheld.
    """
    if not isinstance(call, dict):
        return _redact_secret_keys(call, strings=True)
    return {key: _redact_call_field(key, value) for key, value in call.items()}


def redact_tool_output(content: str) -> str:
    """Redact a tool result: JSON, a truncated result's wrapper, or plain text.

    A result is JSON unless it is plain text, so one that starts like JSON but
    can't be read (cut to size, or too large or deep to parse) is withheld,
    and so is plain text that quotes a secret-shaped key.
    """
    return _redact_text(content, 0, strict=True)


def redact_held_call_result(content: str, *, held: bool = False) -> str:
    """Redact the tool output inside a held call's late result; any other user
    row is the user's own words and is returned as written.

    A row is a late result when its metadata says so (*held*) or when it opens
    with the wrapper's tag.  One that isn't laid out exactly as the gate
    writes it can't be split into the wrapper and the output, so its whole
    body is withheld.
    """
    if not held and not content.lstrip().startswith(_HELD_RESULT_TAG):
        return content
    matched = _HELD_RESULT_RE.fullmatch(content)
    if matched is None:
        return _HELD_RESULT_WITHHELD
    opening, output, closing = matched.groups()
    return f"{opening}{redact_tool_output(output)}{closing}"


def _redact_call_field(key: str, value: Any, in_function: bool = False) -> Any:
    if _is_secret_key(key):
        return _REDACTED
    if key == "arguments" and isinstance(value, str):
        return _redact_arguments(value)
    if key == "function" and isinstance(value, dict) and not in_function:
        return {k: _redact_call_field(k, v, in_function=True) for k, v in value.items()}
    return _redact_secret_keys(value, 1, strings=True)


def _redact_arguments(arguments: str) -> str:
    if not arguments.strip():
        return arguments
    try:
        parsed, repeats_a_key = _load_json(arguments)
    except (ValueError, RecursionError):
        return _REDACTED
    return _redact_parsed(arguments, parsed, repeats_a_key, 1)


def _redact_secret_keys(value: Any, depth: int = 0, *, strings: bool = False) -> Any:
    """Recursively walk *value* and redact secret-shaped subtrees.

    Returns a new structure; never mutates the input.  When a dict key
    matches a secret-shaped hint, the ENTIRE subtree under that key is
    replaced with the ``_REDACTED`` sentinel — including lists and
    nested dicts.  Pre-fix this only redacted string leaves at the
    immediate level, so payloads like ``{"api_keys": ["sk-1", "sk-2"]}``
    or ``{"auth": {"bearer": "tok"}}`` leaked through because the
    inner leaves had no secret-shaped key wrapping them.  Lose the
    (rare, low-value) ability to surface bool/int metadata under a
    secret-shaped key — the sanitizer is the load-bearing security
    boundary for the public viewer.

    Whatever lies deeper than ``_MAX_DEPTH`` is withheld.  With *strings*,
    a string is read for the JSON it may carry (:func:`_redact_text`), and
    what a read returns from one of Otto's working files is withheld
    (:func:`_reads_a_working_file`); the approval card's payload leaves
    strings as written, so it shows what the call will send.
    """
    if depth > _MAX_DEPTH:
        return _REDACTED
    if isinstance(value, dict):
        read_back = _READ_BACK_KEYS if strings and _reads_a_working_file(value) else ()
        return {
            k: (
                _REDACTED
                if _is_secret_key(k) or (k in read_back and isinstance(v, str))
                else _redact_secret_keys(v, depth + 1, strings=strings)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_keys(item, depth + 1, strings=strings) for item in value]
    if strings and isinstance(value, str):
        return _redact_text(value, depth + 1, strict=False)
    return value


def _reads_a_working_file(result: dict) -> bool:
    """Whether *result* names one of Otto's own working files as its ``path``,
    such as a large tool output saved in full.  What a read returns from one
    is a piece of an output whose stored preview is withheld, and a piece
    can hold a value without the key that would get it redacted."""
    path, name = result.get("path"), result.get("name")
    return isinstance(path, str) and is_internal_workspace_file(
        name=name if isinstance(name, str) else "", path=path
    )


def _redact_text(text: str, depth: int, *, strict: bool) -> str:
    """Redact the JSON a string carries, if any.

    A string that starts like JSON but doesn't parse is withheld when
    *strict*; anywhere else it may be prose that merely starts with a
    bracket.  JSON that was cut in the middle, or is too large or too deeply
    nested to read, is withheld either way, and so is text that doesn't
    parse but quotes a secret-shaped key (:func:`_names_a_secret_key`).
    """
    if truncated := _TRUNCATED_RESULT_RE.match(text):
        return _without_preview(truncated.group(1))
    if not _JSON_CONTAINER_RE.match(text):
        return _REDACTED if _names_a_secret_key(text) else text
    if depth > _MAX_DEPTH or _cut_in_the_middle(text):
        return _REDACTED
    try:
        parsed, repeats_a_key = _load_json(text)
    except json.JSONDecodeError:
        return _REDACTED if strict or _names_a_secret_key(text) else text
    except (ValueError, RecursionError):
        return _REDACTED
    return _redact_parsed(text, parsed, repeats_a_key, depth)


def _names_a_secret_key(text: str) -> bool:
    """Whether *text*, which didn't parse, quotes a secret-shaped key before a
    colon.  Then it is a piece of JSON or a dict's repr, such as a window or
    preview of a saved output, and with no parse to say where each value
    ends, the secret can't be cut out of it on its own."""
    return any(
        _is_secret_key(found.group(1)) for found in _QUOTED_KEY_RE.finditer(text)
    )


def _cut_in_the_middle(text: str) -> bool:
    """Whether *text* is a longer text cut down by ``truncate``: its marker
    stands where the removed middle was, with as much kept on either side."""
    return any(
        0 <= marker.start() - (len(text) - marker.end()) <= 1
        for marker in _CUT_MARKER_RE.finditer(text)
    )


def _load_json(text: str) -> tuple[Any, bool]:
    """Parse *text*, and say whether any object in it repeats a key."""
    repeats: list[bool] = []

    def build(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        built = dict(pairs)
        if len(built) < len(pairs):
            repeats.append(True)
        return built

    return json.loads(text, object_pairs_hook=build), bool(repeats)


def _redact_parsed(text: str, parsed: Any, repeats_a_key: bool, depth: int) -> str:
    """*text* as stored when nothing in it needs redacting, so ordinary output
    keeps its layout; otherwise the redacted JSON.

    A repeated key forces the rewrite: parsing keeps only its last value, so
    the stored text could hold an earlier one the walk never saw.
    """
    redacted = _redact_secret_keys(parsed, depth, strings=True)
    if not repeats_a_key and redacted == parsed:
        return text
    return json.dumps(redacted)


def _without_preview(total_chars: str | None) -> str:
    """A truncated result's wrapper, keeping only the size it reports."""
    if total_chars is None:
        return _REDACTED
    return (
        f"<tool-output-truncated total_chars={total_chars}>\n"
        f"{_REDACTED}\n"
        "</tool-output-truncated>"
    )


def _is_secret_key(key: str) -> bool:
    # One spelling for every word boundary, so ``X-Api-Key`` and ``api key``
    # match the two-word hints the way ``api_key`` and ``apiKey`` do.
    normalized_key = _CAMEL_CASE_BOUNDARY_RE.sub("_", key)
    normalized_key = _NON_ALNUM_RE.sub("_", normalized_key)
    return _SECRET_KEY_RE.search(normalized_key) is not None
