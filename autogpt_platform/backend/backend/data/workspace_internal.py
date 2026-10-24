"""Which workspace files are Otto's own working files rather than the user's.

Otto keeps some files in the workspace only so they outlive a summary or a
sandbox: large tool outputs it reads back, the build scratchpad its prompt
asks for, the browser session it restores. The skills registry keeps every
skill package there too (SKILL.md, its references, its LICENSE), installed
when an expert is hired or a skill is saved. The user never asked for any of
them as files, so they must not be listed with what the user did ask for.

``WorkspaceManager.write_file`` asks :func:`is_internal_workspace_file` once
per write and stores the answer on the row (``UserWorkspaceFile.isInternal``),
so every writer is covered without knowing about it. Writers that only make
internal files say so with ``purpose`` in the metadata; the reserved names and
directories cover files whose writer cannot (the model names
``build_state.json`` itself), and a skill folder covers every file in it. The
skills registry publishes packages through its own transaction
(``workspace_skill.publish_workspace_skill_file``), which asks the same
question for each row it creates. The migrations that added the column and
marked skill files apply the same rule, in SQL, to rows written before them.

The rule only ever hides what it recognises: an unknown file is a deliverable.
"""

import re
from typing import Any

from backend.data.workspace_scope import is_skill_package_path

TOOL_OUTPUT_PURPOSE = "tool-output"
WORKING_FILE_PURPOSE = "working-file"
_INTERNAL_PURPOSES = frozenset({TOOL_OUTPUT_PURPOSE, WORKING_FILE_PURPOSE})

WORKING_FILE_METADATA: dict[str, Any] = {"purpose": WORKING_FILE_PURPOSE}

# The scratchpad the system prompt asks for on multi-step builds, and the
# browser session agent_browser saves between turns.
_RESERVED_WORKING_FILE_NAMES = frozenset({"build_state.json", "._browser_state.json"})

# Where large tool outputs are parked, directly under a session's root. A
# user's own "tool-outputs" folder anywhere else is theirs.
_SESSION_TOOL_OUTPUT_PATH = re.compile(
    r"^/sessions/[^/]+/tool-(?:outputs|results)/[^/]+$", re.IGNORECASE
)

_USER_UPLOAD_ORIGIN = "user-upload"


def is_internal_workspace_file(
    *, name: str, path: str, metadata: dict[str, Any] | None = None
) -> bool:
    metadata = metadata or {}
    if metadata.get("origin") == _USER_UPLOAD_ORIGIN:
        return False
    if is_skill_package_path(path):
        return True
    purpose = metadata.get("purpose")
    if purpose is not None:
        return purpose in _INTERNAL_PURPOSES
    if _SESSION_TOOL_OUTPUT_PATH.match(path):
        return True
    return name.lower() in _RESERVED_WORKING_FILE_NAMES
