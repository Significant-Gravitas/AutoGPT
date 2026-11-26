"""GitHub requests for the skills catalog, tried again through a blip.

The catalog is downloaded unattended: by a deploy right after its migrations,
and by ``publish_skills`` when a self-hosted stack, or CI's, starts. A single
504 from GitHub there would leave that environment without the catalog's
skills and experts until the job ran again. So a request rides out a failure
that a later attempt may not meet: a connection that broke, was refused or
timed out, a 5xx, a 429, and a 403 that says when to come back (GitHub's
secondary rate limit). Anything else (a 404 for a ref GitHub doesn't know, a
401, any other 403) is the same the next time and fails at once. When the
attempts or the wait budget run out, the last failure is raised as it came.
"""

from typing import cast

import httpx
from httpx import Client
from tenacity import RetryCallState, wait_exponential_jitter
from tenacity.stop import stop_base

from backend.util.retry import create_retry_decorator

# Six attempts, about 3, 6, 12, 24 and 48 s apart (each plus up to 1 s of
# jitter), so a blip has a minute and a half to pass. A Retry-After is waited
# out as asked, as long as the waits stay within the budget in all; one that
# would take them past it ends the retries at once.
ATTEMPTS = 6
WAIT_BUDGET_S = 120.0
_BACKOFF = wait_exponential_jitter(initial=3)


def github_get(url: str, *, headers: dict[str, str], timeout: float) -> httpx.Response:
    """GET *url*, following redirects. An error status raises
    ``httpx.HTTPStatusError``, as ``raise_for_status`` does."""
    # Built per call, as the service client's retries are, so a test that
    # stands in for ``backend.util.retry``'s sleep reaches it.
    retrying = create_retry_decorator(
        max_attempts=ATTEMPTS,
        context="Skills catalog",
        retry_if=_is_transient,
        wait=_wait,
        also_stop=_WaitBudget(WAIT_BUDGET_S),
    )
    # The decorator is typed for async functions too; this one is sync.
    return cast(httpx.Response, retrying(_get_once)(url, headers, timeout))


def _get_once(url: str, headers: dict[str, str], timeout: float) -> httpx.Response:
    # ``Client`` is this module's own name for httpx's, so a test can hand it
    # a transport that answers as GitHub would.
    with Client(follow_redirects=True, timeout=timeout) as client:
        response = client.get(url, headers=headers)
    response.raise_for_status()
    return response


def _is_transient(error: BaseException) -> bool:
    """Whether a later attempt may not meet *error*."""
    if isinstance(
        error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
    ):
        return True
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    status = error.response.status_code
    if status == 403:
        return _retry_after(error.response) is not None
    return status == 429 or status >= 500


def _wait(retry_state: RetryCallState) -> float:
    """The Retry-After the failure came with, else the backoff."""
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None else None
    if isinstance(error, httpx.HTTPStatusError):
        asked = _retry_after(error.response)
        if asked is not None:
            return asked
    return _BACKOFF(retry_state)


def _retry_after(response: httpx.Response) -> float | None:
    """The seconds a ``Retry-After`` asks for, or None. GitHub sends whole
    seconds; an HTTP-date counts as none."""
    value = response.headers.get("retry-after", "").strip()
    return float(value) if value.isascii() and value.isdigit() else None


class _WaitBudget(stop_base):
    """Stops before a wait that would take the waits past *seconds* in all."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self, retry_state: RetryCallState) -> bool:
        return retry_state.idle_for + retry_state.upcoming_sleep > self.seconds
