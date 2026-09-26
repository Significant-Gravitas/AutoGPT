from enum import Enum
from typing import Any

from backend.sdk import APIKeyCredentials, Requests

CONDUCTOR_API_URL = "https://api.conductor.build"
API_V0 = f"{CONDUCTOR_API_URL}/v0"

# A block-level wall-clock cap for the send-and-wait blocks. The executor
# wraps `run` in `wait_for(execution_timeout_seconds)`, so the block's own
# `timeout_seconds` input is bounded below this.
MAX_WAIT_SECONDS = 2 * 60 * 60

# The API serves at most this many rows per list request (larger `limit`
# values are clamped server-side) and defaults to a much smaller page, so
# every listing that wants more than a handful of rows must page explicitly.
PAGE_SIZE = 100

# Throttled/5xx read responses are retried, but only a few times with short
# back-off: the wait loops enforce their own wall-clock deadline and an
# open-ended retry would silently outlive it.
RETRY_MAX_ATTEMPTS = 4
RETRY_MAX_WAIT_SECONDS = 10.0


class ConductorAgent(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"
    CURSOR = "cursor"
    ACP = "acp"


class ConductorEffort(str, Enum):
    DEFAULT = ""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    ULTRA = "ultra"


class WorkspaceState(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    SLEEPING = "sleeping"
    ARCHIVED = "archived"
    DELETED = "deleted"
    UPDATING = "updating"
    UNSTARTED = "unstarted"


class WorkspaceAction(str, Enum):
    RENAME = "rename"
    ARCHIVE = "archive"
    UNARCHIVE = "unarchive"
    SLEEP = "sleep"
    SHARE_PREVIEW = "share_preview"
    STOP_PREVIEW = "stop_preview"
    MOVE_TO_SECTION = "move_to_section"


class SessionAction(str, Enum):
    RENAME = "rename"
    CANCEL = "cancel"
    ARCHIVE = "archive"


class RoutineAction(str, Enum):
    CREATE = "create"
    ROTATE_WEBHOOK_URL = "rotate_webhook_url"


class SectionAction(str, Enum):
    CREATE = "create"
    DELETE = "delete"


def clean(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop None and empty-string values.

    The API declares `additionalProperties: false` and `minLength: 1` on most
    string fields, so an unset optional must be omitted rather than sent blank.
    Enum members are unwrapped to their value.
    """
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Enum):
            value = value.value
        if value is None or value == "":
            continue
        cleaned[key] = value
    return cleaned


class ConductorClient:
    """Thin async client for api.conductor.build.

    Every route lives under /v0 except `GET /me`, which is served at the API
    root. Errors carry a `userMessage` field which is surfaced verbatim.
    """

    def __init__(self, credentials: APIKeyCredentials):
        self.requests = _requests(credentials, RETRY_MAX_ATTEMPTS)
        self.mutation_requests = _requests(credentials, 1)

    async def _call(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # aiohttp wants repeated keys as a list of pairs, not a list value.
        query: list[tuple[str, Any]] = []
        for key, value in (params or {}).items():
            if value is None or value == "" or value == []:
                continue
            values = value if isinstance(value, list) else [value]
            query.extend((key, _query_value(v)) for v in values)
        requests = self.requests if method == "GET" else self.mutation_requests
        response = await requests.request(
            method, url, json=json_body, params=query or None
        )
        if not response.ok:
            raise ValueError(_error_message(response))
        try:
            body = response.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {"data": body}

    # --- account -----------------------------------------------------------

    async def get_me(self) -> dict[str, Any]:
        return await self._call("GET", f"{CONDUCTOR_API_URL}/me")

    async def list_projects(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        return await self._call(
            "GET", f"{API_V0}/projects", params={"limit": limit, "offset": offset}
        )

    async def get_project(self, project_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/projects/{project_id}")

    async def list_sections(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        return await self._call(
            "GET", f"{API_V0}/sections", params={"limit": limit, "offset": offset}
        )

    async def create_section(self, name: str, emoji: str) -> dict[str, Any]:
        return await self._call(
            "POST",
            f"{API_V0}/sections",
            json_body=clean({"name": name, "emoji": emoji}),
        )

    async def delete_section(self, section_id: str) -> dict[str, Any]:
        return await self._call("DELETE", f"{API_V0}/sections/{section_id}")

    async def list_routines(
        self, limit: int | None = None, offset: int | None = None
    ) -> dict[str, Any]:
        return await self._call(
            "GET", f"{API_V0}/routines", params={"limit": limit, "offset": offset}
        )

    async def create_routine(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", f"{API_V0}/routines", json_body=payload)

    async def rotate_routine_secret(self, routine_id: str) -> dict[str, Any]:
        return await self._call(
            "POST", f"{API_V0}/routines/{routine_id}/rotate-secret", json_body={}
        )

    # --- workspaces --------------------------------------------------------

    async def list_workspaces(
        self, params: dict[str, Any], project_id: str = ""
    ) -> dict[str, Any]:
        url = (
            f"{API_V0}/projects/{project_id}/workspaces"
            if project_id
            else f"{API_V0}/workspaces"
        )
        return await self._call("GET", url, params=params)

    async def create_workspace(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", f"{API_V0}/workspaces", json_body=payload)

    async def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/workspaces/{workspace_id}")

    async def workspace_status(self, workspace_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/workspaces/{workspace_id}/status")

    async def workspace_sessions(
        self,
        workspace_id: str,
        include_archived: bool,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            "GET",
            f"{API_V0}/workspaces/{workspace_id}/sessions",
            params={
                "includeArchived": include_archived,
                "limit": limit,
                "offset": offset,
            },
        )

    async def rename_workspace(self, workspace_id: str, name: str) -> dict[str, Any]:
        return await self._call(
            "POST",
            f"{API_V0}/workspaces/{workspace_id}/rename",
            json_body={"name": name},
        )

    async def workspace_lifecycle(self, workspace_id: str, verb: str) -> dict[str, Any]:
        """verb is one of archive, unarchive, sleep."""
        return await self._call(
            "POST", f"{API_V0}/workspaces/{workspace_id}/{verb}", json_body={}
        )

    async def get_preview(self, workspace_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/workspaces/{workspace_id}/preview")

    async def share_preview(self, workspace_id: str, port: int) -> dict[str, Any]:
        return await self._call(
            "PUT",
            f"{API_V0}/workspaces/{workspace_id}/preview",
            json_body={"port": port},
        )

    async def stop_preview(self, workspace_id: str) -> dict[str, Any]:
        return await self._call("DELETE", f"{API_V0}/workspaces/{workspace_id}/preview")

    async def set_workspace_section(
        self, workspace_id: str, section_id: str | None
    ) -> dict[str, Any]:
        return await self._call(
            "PUT",
            f"{API_V0}/workspaces/{workspace_id}/section",
            json_body={"sectionId": section_id},
        )

    # --- sessions ----------------------------------------------------------

    async def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", f"{API_V0}/sessions", json_body=payload)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/sessions/{session_id}")

    async def session_status(self, session_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/sessions/{session_id}/status")

    async def rename_session(self, session_id: str, name: str) -> dict[str, Any]:
        return await self._call(
            "POST", f"{API_V0}/sessions/{session_id}/rename", json_body={"name": name}
        )

    async def cancel_session(self, session_id: str) -> dict[str, Any]:
        return await self._call(
            "POST", f"{API_V0}/sessions/{session_id}/cancel", json_body={}
        )

    async def archive_session(self, session_id: str) -> dict[str, Any]:
        return await self._call(
            "POST", f"{API_V0}/sessions/{session_id}/archive", json_body={}
        )

    async def send_message(self, session_id: str, message: str) -> dict[str, Any]:
        return await self._call(
            "POST",
            f"{API_V0}/sessions/{session_id}/messages",
            json_body={"message": message},
        )

    async def list_messages(
        self,
        session_id: str,
        after: str = "",
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """One page of a session's transcript, oldest first.

        `after` is an exclusive row-id cursor and cannot be combined with
        `offset`. Pages are clamped to PAGE_SIZE rows server-side.
        """
        return await self._call(
            "GET",
            f"{API_V0}/sessions/{session_id}/messages",
            params={"after": after, "limit": limit, "offset": offset},
        )

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return await self._call("GET", f"{API_V0}/messages/{message_id}")


def _requests(credentials: APIKeyCredentials, attempts: int) -> Requests:
    return Requests(
        trusted_origins=[CONDUCTOR_API_URL],
        raise_for_status=False,
        extra_headers={
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}"
        },
        retry_max_attempts=attempts,
        retry_max_wait=RETRY_MAX_WAIT_SECONDS,
    )


def _query_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return value.value
    return value


def _error_message(response: Any) -> str:
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(body.get("userMessage") or body.get("message") or "")
    except Exception:
        pass
    if not detail:
        try:
            detail = response.text()[:300]
        except Exception:
            detail = ""
    prefix = f"Conductor API error (HTTP {response.status})"
    return f"{prefix}: {detail}" if detail else prefix
