"""The HTTP requests a provider SDK sends for one block LLM call (#16173).

A block's LLM call runs under one timeout, and that timeout also covers every
retry the SDK makes on its own and the backoff between them. To tell whether
the timeout cut off a request the provider was still generating, the spend
alert needs each request's start and the status it got back. The block-side
``llm_call`` sets a log in ``current_http_attempts``, and the httpx hooks on the
provider clients write to it. Other callers set none, so the hooks do nothing.
"""

from collections.abc import Callable
from contextvars import ContextVar

import httpx
from pydantic import BaseModel, Field


class HttpAttemptLog(BaseModel):
    """Each HTTP request an SDK sent for one block LLM call, retries included."""

    clock: Callable[[], float]
    starts: list[float] = Field(default_factory=list)
    # One per request: the status of its response, None while it has none.
    # A request that never gets one (a dropped connection) keeps None.
    statuses: list[int | None] = Field(default_factory=list)

    def request_sent(self) -> None:
        self.starts.append(self.clock())
        self.statuses.append(None)

    def response_received(self, status: int) -> None:
        # The SDK sends one request at a time, so a response answers the last.
        self.statuses[-1] = status


current_http_attempts: ContextVar[HttpAttemptLog | None] = ContextVar(
    "current_http_attempts", default=None
)


def attempt_logging_http_client(
    default_client: Callable[..., httpx.AsyncClient],
) -> httpx.AsyncClient:
    """The SDK's default httpx client, with hooks that see every SDK retry.

    An SDK never closes a client it was handed, so the caller must.
    """
    return default_client(
        event_hooks={
            "request": [record_http_request],
            "response": [record_http_response],
        }
    )


async def record_http_request(request: httpx.Request) -> None:
    log = current_http_attempts.get()
    if log is not None:
        log.request_sent()


async def record_http_response(response: httpx.Response) -> None:
    log = current_http_attempts.get()
    if log is not None:
        log.response_received(response.status_code)
