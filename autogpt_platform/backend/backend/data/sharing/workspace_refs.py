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

_WORKSPACE_PREFIX = "workspace://"

# Match ``file_id`` followed by either ``=`` (Attached-files block) or
# ``":"`` / ``": "`` (JSON serialised tool output).  Anchored on the
# ``file_id`` literal so the UUID alone (e.g. mid-narrative) is never
# pulled out as a reference.  The ``\b`` after the UUID prevents
# trailing characters from sneaking in.
_FILE_ID_RE = re.compile(
    r'(?:file_id=|"file_id"\s*:\s*")'
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)


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
        if value.startswith(_WORKSPACE_PREFIX):
            raw = value.removeprefix(_WORKSPACE_PREFIX)
            file_ref = raw.split("#", 1)[0] if "#" in raw else raw
            # Reject leading slashes — those would denote a path under
            # the workspace, not a file ID, and our allowlist keys on
            # file ID only.
            if file_ref and not file_ref.startswith("/"):
                sink.add(file_ref)
        # Also pick up ``file_id=<uuid>`` tokens from [Attached files]
        # blocks — these show up inside arbitrary message content, not
        # as standalone leaves, so the substring scan is required.
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
