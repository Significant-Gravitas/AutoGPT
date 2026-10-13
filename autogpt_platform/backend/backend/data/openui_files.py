"""Saved OpenUI views: workspace files whose body is an OpenUI program.

The file's metadata names the view (``{"openui": {"viewId", "originSessionId"}}``)
and the user's edits live in a ``ChatUIDraft`` row tied to the file. These
run in the CoPilot executor too, so they are exposed on the DatabaseManager.
"""

import posixpath
from typing import Optional

from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction
from backend.data.workspace import OPENUI_EXTENSION, OPENUI_MIME_TYPE, WorkspaceFile
from backend.data.workspace_scope import session_path_prefix


def is_openui_name(name: str | None) -> bool:
    return name is not None and name.lower().endswith(OPENUI_EXTENSION)


def openui_filename(name: str) -> str:
    """A plain file name for a view, ending in ``.openui``."""
    base = posixpath.basename(name.strip().replace("\\", "/")).strip()
    if not base or base in {".", ".."}:
        raise ValueError("Give the view file a name, e.g. trip-plan.openui.")
    return base if is_openui_name(base) else f"{base}{OPENUI_EXTENSION}"


def is_openui_file(file: WorkspaceFile) -> bool:
    return file.mime_type == OPENUI_MIME_TYPE or is_openui_name(file.name)


def openui_metadata(base: dict, view_id: str, session_id: str) -> dict:
    """``base`` with the view's identity, naming ``session_id`` as the chat
    that last wrote it."""
    return {**base, "openui": {"viewId": view_id, "originSessionId": session_id}}


def openui_view_id(metadata: dict) -> str | None:
    return _openui_field(metadata, "viewId")


def openui_origin_session_id(metadata: dict) -> str | None:
    return _openui_field(metadata, "originSessionId")


def _openui_field(metadata: dict, key: str) -> str | None:
    openui = metadata.get("openui")
    if not isinstance(openui, dict):
        return None
    value = openui.get(key)
    return value if isinstance(value, str) and value else None


async def find_openui_view_file(
    user_id: str, session_id: str, view_id: str
) -> Optional[str]:
    """ID of the user's live file that holds ``view_id`` for this chat.

    A file belongs to the chat when it lives in the chat's folder or the chat
    last wrote it. Oldest first, so concurrent saves agree on one file.
    """
    rows = await query_raw_with_schema(
        'SELECT f.id FROM {schema_prefix}"UserWorkspaceFile" f '
        'JOIN {schema_prefix}"UserWorkspace" w ON w.id = f."workspaceId" '
        'WHERE w."userId" = $1 AND f."isDeleted" = false '
        "AND f.metadata->'openui'->>'viewId' = $2 "
        "AND (f.metadata->'openui'->>'originSessionId' = $3 "
        "OR starts_with(f.path, $4)) "
        'ORDER BY f."createdAt", f.id LIMIT 1',
        user_id,
        view_id,
        session_id,
        session_path_prefix(session_id),
    )
    return rows[0]["id"] if rows else None


async def get_openui_file_state(user_id: str, file_id: str) -> Optional[dict]:
    """The user's current edits to a saved view, or None when there are none."""
    rows = await query_raw_with_schema(
        'SELECT d.state FROM {schema_prefix}"ChatUIDraft" d '
        'JOIN {schema_prefix}"UserWorkspaceFile" f ON f.id = d."fileId" '
        'JOIN {schema_prefix}"UserWorkspace" w ON w.id = f."workspaceId" '
        'WHERE d."fileId" = $1 AND w."userId" = $2 AND f."isDeleted" = false',
        file_id,
        user_id,
    )
    state = rows[0]["state"] if rows else None
    return state if isinstance(state, dict) else None


async def move_chat_edits_to_file(
    user_id: str, session_id: str, view_id: str, file_id: str
) -> bool:
    """Move the chat's unsaved edits to ``view_id`` onto the user's new view
    file, so they follow the view. False when there were none to move.

    Locks the chat first: an edit write holds the same lock while it checks
    that the view has no file yet, so no edit lands in the chat after this.
    """
    async with transaction() as tx:
        owned = await query_raw_with_schema(
            'SELECT id FROM {schema_prefix}"ChatSession" '
            'WHERE id = $1 AND "userId" = $2 FOR UPDATE',
            session_id,
            user_id,
            client=tx,
        )
        if not owned:
            return False
        moved = await execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatUIDraft" SET "sessionId" = NULL, '
            '"fileId" = $1, "updatedAt" = NOW() '
            'WHERE "sessionId" = $2 AND "viewId" = $3 AND EXISTS ('
            'SELECT 1 FROM {schema_prefix}"UserWorkspaceFile" f '
            'JOIN {schema_prefix}"UserWorkspace" w ON w.id = f."workspaceId" '
            'WHERE f.id = $1 AND w."userId" = $4)',
            file_id,
            session_id,
            view_id,
            user_id,
            client=tx,
        )
    return moved > 0
