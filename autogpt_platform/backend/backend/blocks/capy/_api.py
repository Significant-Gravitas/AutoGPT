"""Async client for the Capy public API v1 (https://docs.capy.ai/api-reference)."""

import uuid
from typing import Any, Awaitable, Callable, Optional, TypeVar

from backend.sdk import APIKeyCredentials, Requests
from backend.util.request import Response

from ._models import ModelRoute, is_linked_route, rejection_hint, resolve_model_id
from ._types import (
    Message,
    MessagePage,
    MessageReceipt,
    Project,
    ReviewRound,
    ReviewStarted,
    Task,
    Thread,
)

API_URL = "https://api.capy.ai/api/v1"

# Transcript cursors are 26-character event ULIDs. The largest one sorts after
# every real event, so paging backward from it returns the newest page.
_NEWEST_EVENT_CURSOR = "7" + "Z" * 25

# How many forward pages to walk when looking for the newest messages without
# the backward cursor. Bounds the cost on very long threads; a longer one
# fails instead of returning an older page as the newest.
_MAX_FORWARD_PAGES = 50


T = TypeVar("T")


class CapyAPIError(RuntimeError):
    """A non-2xx answer from Capy, carrying its ``_tag`` (e.g. ``capy/Forbidden``).

    A rejected model also carries Capy's ``rejection`` (``disconnected`` or
    ``not_connected``) and the linked ``service`` it names.
    """

    def __init__(
        self,
        status: int,
        tag: str,
        message: str,
        *,
        rejection: str | None = None,
        service: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.tag = tag
        self.rejection = rejection
        self.service = service


async def with_capy_balance_fallback(
    call: Callable[[str], Awaitable[T]], model_id: str, fallback: bool
) -> tuple[T, str]:
    """Run ``call`` with ``model_id``; if its linked provider is unavailable and
    ``fallback`` is on, run the same model on the Capy balance instead.

    Returns the result and the model ID that was actually used.
    """
    try:
        return await call(model_id), model_id
    except CapyAPIError as exc:
        if not (
            fallback
            and exc.tag == "ModelSelection.Rejected"
            and is_linked_route(model_id)
        ):
            raise
    balance_id = resolve_model_id(model_id, ModelRoute.CAPY_BALANCE)
    return await call(balance_id), balance_id


class CapyClient:
    def __init__(self, credentials: APIKeyCredentials):
        self.requests = Requests(
            trusted_origins=[API_URL],
            raise_for_status=False,
            extra_headers={
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
    ) -> Any:
        response = await self.requests.request(
            method,
            f"{API_URL}{path}",
            params={k: v for k, v in (params or {}).items() if v not in (None, "")},
            json=body,
        )
        if not response.ok:
            raise _error(response)
        if not response.content:
            return None
        return response.json()

    # --- Projects ---------------------------------------------------------

    async def list_projects(self) -> list[Project]:
        data = await self._request("GET", "/projects")
        return [Project.model_validate(p) for p in data.get("items", [])]

    async def get_project(self, project_id: str) -> Project:
        return Project.model_validate(
            await self._request("GET", f"/projects/{project_id}")
        )

    # --- Threads ----------------------------------------------------------

    async def create_thread(
        self,
        *,
        project_id: str,
        message: str,
        title: str = "",
        model_id: str = "",
        reasoning: str = "",
        machine_size: str = "",
        request_id: str = "",
    ) -> Thread:
        body: dict[str, Any] = {
            # Capy dedupes creates on requestId, so the retry layer in
            # Requests can never start a second run for one block call.
            "requestId": request_id or str(uuid.uuid4()),
            "projectId": project_id,
            "message": message,
        }
        if title:
            body["title"] = title
        if model := _model_selection(model_id, reasoning):
            body["model"] = model
        if machine_size:
            body["machineSize"] = machine_size
        return Thread.model_validate(await self._request("POST", "/threads", body=body))

    async def get_thread(self, thread_id: str) -> Thread:
        return Thread.model_validate(
            await self._request("GET", f"/threads/{thread_id}")
        )

    async def list_threads(
        self, project_id: str, limit: int, cursor: str = ""
    ) -> tuple[list[Thread], Optional[str]]:
        data = await self._request(
            "GET",
            "/threads",
            params={"projectId": project_id, "limit": limit, "cursor": cursor},
        )
        threads = [Thread.model_validate(t) for t in data.get("items", [])]
        return threads, data.get("cursor")

    async def archive_thread(self, thread_id: str) -> Thread:
        return Thread.model_validate(
            await self._request("POST", f"/threads/{thread_id}/archive")
        )

    # --- Messages ---------------------------------------------------------

    async def list_messages(
        self,
        thread_id: str,
        *,
        limit: int,
        after: str = "",
        before: str = "",
    ) -> MessagePage:
        data = await self._request(
            "GET",
            f"/threads/{thread_id}/messages",
            params={"limit": limit, "after": after, "before": before},
        )
        return MessagePage.model_validate(data)

    async def newest_messages(self, thread_id: str, limit: int) -> MessagePage:
        """The newest ``limit`` transcript entries, oldest first.

        Tries the backward cursor first; if Capy ever rejects it, walks the
        transcript forward and keeps the tail.
        """
        try:
            return await self.list_messages(
                thread_id, limit=limit, before=_NEWEST_EVENT_CURSOR
            )
        except CapyAPIError as exc:
            if exc.tag != "capy/InvalidRequest":
                raise
        tail: list[Message] = []
        cursor = ""
        for _ in range(_MAX_FORWARD_PAGES):
            page = await self.list_messages(thread_id, limit=100, after=cursor)
            tail = (tail + page.items)[-limit:]
            if not page.cursor or page.cursor == cursor:
                return MessagePage(items=tail, cursor=page.cursor or cursor or None)
            cursor = page.cursor
        # Returning the tail now would pass an older page off as the newest.
        raise RuntimeError(
            "Capy rejected the newest-first transcript cursor, and the "
            f"transcript runs past {_MAX_FORWARD_PAGES * 100} entries, so its "
            "newest messages can't be reached by reading forward"
        )

    async def send_message(
        self,
        thread_id: str,
        *,
        text: str,
        delivery: str,
        model_id: str = "",
        reasoning: str = "",
    ) -> MessageReceipt:
        body: dict[str, Any] = {"text": text, "delivery": delivery}
        if model := _model_selection(model_id, reasoning):
            body["model"] = model
        return MessageReceipt.model_validate(
            await self._request("POST", f"/threads/{thread_id}/message", body=body)
        )

    async def interrupt_thread(self, thread_id: str) -> MessageReceipt:
        return MessageReceipt.model_validate(
            await self._request("POST", f"/threads/{thread_id}/interrupt")
        )

    # --- Tasks ------------------------------------------------------------

    async def list_tasks(
        self, thread_id: str, limit: int, after: str = ""
    ) -> tuple[list[Task], Optional[str]]:
        data = await self._request(
            "GET",
            f"/threads/{thread_id}/tasks",
            params={"limit": limit, "after": after},
        )
        return [Task.model_validate(t) for t in data.get("items", [])], data.get(
            "cursor"
        )

    # --- Reviews ----------------------------------------------------------

    async def start_review(
        self,
        *,
        repo: str,
        pr_number: int,
        idempotency_key: str = "",
        tier: str = "",
        source_thread_id: str = "",
        force_refresh: bool = False,
    ) -> ReviewStarted:
        body: dict[str, Any] = {"repo": repo, "prNumber": pr_number}
        if idempotency_key:
            body["idempotencyKey"] = idempotency_key
        if tier:
            body["tier"] = tier
        if source_thread_id:
            body["sourceThreadId"] = source_thread_id
        if force_refresh:
            body["forceRefresh"] = True
        return ReviewStarted.model_validate(
            await self._request("POST", "/reviews", body=body)
        )

    async def get_review_round(self, request_id: str) -> ReviewRound:
        return ReviewRound.model_validate(
            await self._request("GET", f"/reviews/rounds/{request_id}")
        )

    # --- Usage ------------------------------------------------------------

    async def get_usage(self, from_: str = "", to: str = "") -> dict[str, Any]:
        return await self._request("GET", "/usage", params={"from": from_, "to": to})


def _model_selection(model_id: str, reasoning: str) -> Optional[dict[str, Any]]:
    if not model_id:
        return None
    selection: dict[str, Any] = {"modelId": model_id}
    if reasoning:
        selection["reasoningMode"] = reasoning
    return selection


def _error(response: Response) -> CapyAPIError:
    """Turn Capy's tagged error envelope into a message a person can act on."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        text = response.text()[:200]
        return CapyAPIError(
            response.status, "", f"Capy returned HTTP {response.status}: {text}"
        )

    tag = str(body.get("_tag", ""))
    detail = body.get("message") or body.get("reason") or ""
    hints = {
        "capy/Unauthorized": "the API key is missing, revoked or expired",
        "capy/Forbidden": "the key's principal is not allowed to do this "
        "(a read_only service key, or a project outside its access)",
        "capy/RateLimited": "rate limited; retry after "
        f"{body.get('retryAfterSeconds', '?')} seconds",
        "capy/PaidFeatureUnavailable": "this needs a paid Capy plan "
        f"({body.get('feature', '')})",
        "capy/ProjectNotFound": "no project with that ID is visible to this key",
        "capy/ThreadNotFound": "no thread with that ID is visible to this key",
        "capy/ReviewRoundNotFound": "no review round with that request ID",
        "capy/ReviewRefused": "Capy refused to review this pull request",
        "ModelSelection.Rejected": rejection_hint(
            body.get("rejection"), body.get("service"), str(body.get("modelId", ""))
        ),
    }
    explanation = hints.get(tag, "request failed")
    message = f"Capy {tag or 'error'} (HTTP {response.status}): {explanation}"
    if detail:
        message += f": {detail}"
    if candidates := body.get("candidates"):
        names = ", ".join(str(c.get("entryId", c.get("name"))) for c in candidates)
        message += f". Available models: {names}"
    return CapyAPIError(
        response.status,
        tag,
        message,
        rejection=body.get("rejection"),
        service=body.get("service"),
    )
