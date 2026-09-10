"""Extract workspace file references from arbitrary nested data.

Both execution-output sharing and chat-message sharing need to build an
allowlist of workspace files exposed by a share.  Three reference shapes
appear in real payloads:

1. ``workspace://<uuid>`` URI strings — emitted by
   :func:`backend.util.file.store_media_file`, used by blocks and tool
   outputs.  Anchored prefix match keeps mid-text occurrences intact.
2. ``file_id=<uuid>`` substrings in ``[Attached files]`` blocks the
   copilot appends to user messages (see
   ``backend/copilot/pending_messages.py``).  These show up inside
   ChatMessage.content for user uploads.
3. ``"file_id":"<uuid>"`` JSON tokens in tool-response message content
   (``role="tool"`` rows).  Tool outputs like
   ``workspace_file_written`` serialise the file id inside a JSON
   blob persisted as plain text — the JSON shape uses ``":"`` between
   key and value rather than ``=``, so a separate alternation is
   required.

All three forms are handled by a single recursive walker.
"""

import re
from typing import Any

from pydantic import BaseModel

# Match ``workspace://<id>`` anywhere in a string.  Assistant messages
# embed these URIs inside markdown (``![chart](workspace://uuid#mime)``)
# so a whole-string ``startswith`` check used to miss them — the file
# would render in the chat but 404 in the public viewer because no
# allowlist row existed.  ``re.finditer`` over the full content catches
# both the standalone-URI shape AND the embedded-in-markdown shape.
#
# Char class is lowercase alphanumeric + hyphen (the UUID alphabet).
# Stops at ``/`` so ``workspace:///path/to/file`` cannot escape into a
# path reference, and stops at ``#`` so the optional MIME hint fragment
# is excluded from the captured ID.
_WORKSPACE_URI_RE = re.compile(r"workspace://([a-z0-9-]+)")

# Match ``file_id`` followed by either ``=`` (Attached-files block) or
# ``":"`` / ``": "`` (JSON serialised tool output).  Anchored on the
# ``file_id`` literal so the UUID alone (e.g. mid-narrative) is never
# pulled out as a reference.  The ``\b`` after the UUID prevents
# trailing characters from sneaking in.
_FILE_ID_RE = re.compile(
    r'(?:file_id=|"file_id"\s*:\s*")'
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)

# Full markdown artifact reference the assistant emits — either a file link
# ``[report.csv](workspace://<id>#text/csv)`` or an image embed
# ``![chart](workspace://<id>#image/png)`` (the prompt instructs both forms,
# see ``backend/copilot/prompting.py``). The leading ``!`` is consumed so image
# embeds don't leave a dangling marker once the link is stripped. Builds on the
# same ``workspace://<id>`` grammar as ``_WORKSPACE_URI_RE`` above, but also
# captures the human label and optional ``#mime`` hint — chat-bot delivery needs
# both to attach the file with a sensible filename.
_WORKSPACE_ARTIFACT_LINK_RE = re.compile(
    r"!?\[([^\]]+)\]\(workspace://([a-z0-9-]+)(?:#([^)]*))?\)"
)


class WorkspaceArtifactLink(BaseModel):
    """A ``[name](workspace://id#mime)`` reference parsed out of assistant text."""

    display_name: str
    file_id: str
    mime_hint: str | None = None


def extract_artifact_links(text: str) -> tuple[str, list[WorkspaceArtifactLink]]:
    """Pull ``[name](workspace://id#mime)`` markdown links out of *text*.

    Returns the text with those links removed (surrounding whitespace tidied)
    plus the parsed artifacts in document order. Raw ``workspace://`` URIs are
    useless to a chat-platform user, so callers strip them here and deliver the
    referenced file (or a link to it) separately.
    """
    artifacts: list[WorkspaceArtifactLink] = []

    def _capture(match: re.Match[str]) -> str:
        artifacts.append(
            WorkspaceArtifactLink(
                display_name=match.group(1).strip(),
                file_id=match.group(2),
                mime_hint=match.group(3) or None,
            )
        )
        return ""

    stripped = _WORKSPACE_ARTIFACT_LINK_RE.sub(_capture, text)
    # Clean up the gaps the removed links leave behind so surrounding prose
    # still reads naturally.
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip(), artifacts


def cut_lands_inside_artifact_link(text: str, cut: int) -> int:
    """Move *cut* off any artifact link it would bisect, keeping the link whole.

    Splitting a streamed buffer mid-``[name](workspace://...)`` would leave the
    fragment unrecognisable to :func:`extract_artifact_links`, so the link is
    kept intact. Normally we pull the cut back to the link's start so the link
    travels into the remainder — but if the link sits at the very start of the
    buffer, that yields an empty chunk and stalls the stream (the buffer never
    advances). In that case we cut at the link's end instead, emitting the whole
    link as one chunk. Either way the returned cut makes forward progress.
    """
    for match in _WORKSPACE_ARTIFACT_LINK_RE.finditer(text):
        if match.start() < cut < match.end():
            return match.start() if match.start() > 0 else match.end()
    return cut


def extract_workspace_file_ids(value: Any) -> set[str]:
    """Walk *value* recursively and collect referenced workspace file IDs.

    Accepts any JSON-shaped value: strings, lists, dicts, primitives.
    Non-string leaves are ignored.  Returns the unique set of file IDs
    referenced by either ``workspace://<id>`` URIs or ``file_id=<uuid>``
    tokens (from ``[Attached files]`` blocks).
    """
    file_ids: set[str] = set()
    _scan(value, file_ids)
    return file_ids


def _scan(value: Any, sink: set[str]) -> None:
    if isinstance(value, str):
        # ``workspace://<uuid>`` anywhere in the string — handles both
        # standalone leaves (``["workspace://abc"]``) and URIs embedded
        # in markdown (``![chart](workspace://abc#image/png)``).
        for match in _WORKSPACE_URI_RE.finditer(value):
            sink.add(match.group(1))
        # ``file_id=<uuid>`` tokens from [Attached files] blocks AND
        # ``"file_id":"<uuid>"`` tokens from JSON-serialised tool output.
        for match in _FILE_ID_RE.finditer(value):
            sink.add(match.group(1))
        return
    if isinstance(value, list):
        for item in value:
            _scan(item, sink)
        return
    if isinstance(value, dict):
        for v in value.values():
            _scan(v, sink)
