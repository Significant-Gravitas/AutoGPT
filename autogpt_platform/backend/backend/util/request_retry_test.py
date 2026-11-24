"""A throttle status is retried a bounded number of times, and a redirect
chain shares those attempts (#17130)."""

import asyncio
import socket
import time
from urllib.parse import ParseResult

import pytest
from aiohttp import ClientConnectorError, web

from backend.util import request as request_module
from backend.util.request import (
    DEFAULT_RETRY_MAX_ATTEMPTS,
    HTTPClientError,
    HTTPServerError,
    Requests,
)
from backend.util.request_test_helpers import (
    ENDS_WITHIN_SECONDS,
    RETRYING_CLIENTS,
    local_requests,
    retry_waits,
    scripted,
    serve,
)


@pytest.mark.asyncio
async def test_a_persistent_throttle_status_ends_after_the_default_attempts():
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        response = await asyncio.wait_for(
            local_requests(raise_for_status=False, retry_max_wait=0).get(base),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert response.status == 503
    assert len(served) == DEFAULT_RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_raise_for_status_retries_up_to_its_cap_then_raises_the_last_error():
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        with pytest.raises(HTTPServerError) as raised:
            await asyncio.wait_for(
                local_requests(retry_max_attempts=3, retry_max_wait=0).get(base),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert served == [503, 503, 503]
    assert raised.value.status_code == 503
    assert raised.value.body == b"attempt 3"


@pytest.mark.asyncio
async def test_raise_for_status_without_a_cap_still_makes_one_attempt():
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        with pytest.raises(HTTPServerError):
            await asyncio.wait_for(
                local_requests(retry_max_wait=0).get(base), ENDS_WITHIN_SECONDS
            )
    finally:
        await runner.cleanup()

    assert served == [503]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
@pytest.mark.parametrize(
    "raise_for_status", [True, False], ids=["raising", "status-checked"]
)
async def test_a_capped_client_does_not_retry_a_status_that_is_not_a_throttle(
    status: int, raise_for_status: bool
):
    """Only a throttle status is retried: anything else would be sent again,
    a POST included, for an answer that can't change."""
    app, served = scripted(status)
    runner, base = await serve(app)

    try:
        client = local_requests(
            raise_for_status=raise_for_status, retry_max_attempts=3, retry_max_wait=0
        )
        if raise_for_status:
            with pytest.raises((HTTPClientError, HTTPServerError)) as raised:
                await asyncio.wait_for(client.post(base), ENDS_WITHIN_SECONDS)
            assert raised.value.status_code == status
        else:
            response = await asyncio.wait_for(client.post(base), ENDS_WITHIN_SECONDS)
            assert response.status == status
    finally:
        await runner.cleanup()

    assert served == [status]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raise_for_status", [True, False], ids=["raising", "status-checked"]
)
async def test_an_explicit_cap_above_the_default_is_honoured(raise_for_status: bool):
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        client = local_requests(
            raise_for_status=raise_for_status, retry_max_attempts=5, retry_max_wait=0
        )
        if raise_for_status:
            with pytest.raises(HTTPServerError):
                await asyncio.wait_for(client.get(base), ENDS_WITHIN_SECONDS)
        else:
            response = await asyncio.wait_for(client.get(base), ENDS_WITHIN_SECONDS)
            assert response.status == 503
    finally:
        await runner.cleanup()

    assert served == [503] * 5


@pytest.mark.asyncio
async def test_the_default_attempts_wait_seconds_not_minutes():
    """The default exists so a block against a host that is down fails in
    seconds: its retries wait 1 s, then 2 s, each plus up to 1 s of jitter."""
    waits = await retry_waits(Requests(raise_for_status=False))

    assert len(waits) == DEFAULT_RETRY_MAX_ATTEMPTS - 1
    assert sum(waits) < 5


@pytest.mark.asyncio
async def test_a_throttled_send_waits_its_backoff_before_going_again():
    """Each retry waits its backoff, here capped at 0.2 s, so two retries take
    at least 0.4 s (the bound allows for the clock's rounding)."""
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        started = time.monotonic()
        await asyncio.wait_for(
            local_requests(raise_for_status=False, retry_max_wait=0.2).get(base),
            ENDS_WITHIN_SECONDS,
        )
        elapsed = time.monotonic() - started
    finally:
        await runner.cleanup()

    assert served == [503, 503, 503]
    assert elapsed >= 0.35


@pytest.mark.asyncio
@RETRYING_CLIENTS
async def test_a_connection_error_is_never_retried(client_kwargs: dict):
    attempts: list[str] = []

    def count(url: ParseResult) -> ParseResult:
        attempts.append(url.geturl())
        return url

    # Bound but not listening, so a connection to it is refused.
    with socket.socket() as refusing:
        refusing.bind(("127.0.0.1", 0))
        port = refusing.getsockname()[1]
        with pytest.raises(ClientConnectorError):
            await asyncio.wait_for(
                local_requests(
                    retry_max_wait=0, extra_url_validator=count, **client_kwargs
                ).post(f"http://127.0.0.1:{port}/"),
                ENDS_WITHIN_SECONDS,
            )

    assert attempts == [f"http://127.0.0.1:{port}/"]


@pytest.mark.asyncio
@RETRYING_CLIENTS
async def test_a_throttle_status_then_success_succeeds_on_the_second_attempt(
    client_kwargs: dict,
):
    app, served = scripted(429, 200)
    runner, base = await serve(app)

    try:
        response = await asyncio.wait_for(
            local_requests(retry_max_wait=0, **client_kwargs).get(base),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert served == [429, 200]
    assert response.status == 200
    assert response.text() == "attempt 2"


def _chain(
    scripts: dict[str, list[tuple[int, str | None]]],
) -> tuple[web.Application, dict[str, list[bytes]]]:
    """An app where each path answers its script in turn, repeating the last
    answer for good: a status, and where a redirect leads. It records each
    request's body, per path."""
    received: dict[str, list[bytes]] = {path: [] for path in scripts}

    def answering(path: str):
        async def answer(request: web.Request) -> web.Response:
            received[path].append(await request.read())
            script = scripts[path]
            status, location = script[min(len(received[path]), len(script)) - 1]
            headers = {"Location": location} if location else {}
            return web.Response(status=status, headers=headers)

        return answer

    app = web.Application()
    for path in scripts:
        app.router.add_route("*", path, answering(path))
    return app, received


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, client_kwargs, scripts, outcome, sends",
    [
        pytest.param(
            "GET",
            {"raise_for_status": False},
            {"/a": [(302, "/b")], "/b": [(429, None), (200, None)]},
            200,
            {"/a": 1, "/b": 2},
            id="get-whose-second-hop-is-throttled-once",
        ),
        pytest.param(
            "POST",
            {"raise_for_status": False},
            {"/a": [(307, "/b")], "/b": [(503, None), (200, None)]},
            200,
            {"/a": 1, "/b": 2},
            id="post-307-whose-second-hop-is-throttled-once",
        ),
        pytest.param(
            "POST",
            {"raise_for_status": False},
            {"/a": [(303, "/b")], "/b": [(429, None), (200, None)]},
            200,
            {"/a": 1, "/b": 2},
            id="post-303-whose-target-is-throttled-once",
        ),
        pytest.param(
            "GET",
            {"raise_for_status": False, "retry_max_attempts": 3},
            {"/a": [(302, "/b")], "/b": [(503, None)]},
            503,
            {"/a": 1, "/b": 3},
            id="second-hop-down",
        ),
        pytest.param(
            "GET",
            {"raise_for_status": False, "retry_max_attempts": 3},
            {"/a": [(302, "/b")], "/b": [(302, "/c")], "/c": [(503, None)]},
            503,
            {"/a": 1, "/b": 1, "/c": 3},
            id="third-hop-down",
        ),
        pytest.param(
            "GET",
            {"retry_max_attempts": 3},
            {"/a": [(302, "/b")], "/b": [(503, None)]},
            HTTPServerError,
            {"/a": 1, "/b": 3},
            id="second-hop-down-raising",
        ),
        pytest.param(
            "GET",
            {"raise_for_status": False, "retry_max_attempts": 3},
            {"/a": [(503, None), (302, "/b")], "/b": [(503, None)]},
            503,
            {"/a": 2, "/b": 2},
            id="one-budget-for-the-whole-chain",
        ),
    ],
)
async def test_a_redirect_chain_sends_again_only_the_throttled_hop(
    method: str,
    client_kwargs: dict,
    scripts: dict[str, list[tuple[int, str | None]]],
    outcome: int | type[Exception],
    sends: dict[str, int],
):
    """A throttled hop is sent again on its own: the hops before it are not,
    so a POST is never repeated for a later hop's sake, and the chain shares
    one budget of throttled sends."""
    app, received = _chain(scripts)
    runner, base = await serve(app)

    try:
        sent = local_requests(retry_max_wait=0, **client_kwargs).request(
            method, f"{base}/a", json={"order": 1} if method == "POST" else None
        )
        if isinstance(outcome, int):
            response = await asyncio.wait_for(sent, ENDS_WITHIN_SECONDS)
            assert response.status == outcome
        else:
            with pytest.raises(outcome):
                await asyncio.wait_for(sent, ENDS_WITHIN_SECONDS)
    finally:
        await runner.cleanup()

    assert {path: len(bodies) for path, bodies in received.items()} == sends


# --- The waits between attempts, noted instead of slept ---


@pytest.fixture
def backoff_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits between attempts, through the ``sleep`` the module bound."""
    waits: list[float] = []

    async def note(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(request_module, "sleep", note)
    return waits


@pytest.mark.asyncio
async def test_each_retry_waits_a_doubling_backoff_and_none_follows_the_last_attempt(
    backoff_waits: list[float],
):
    """1 s, 2 s, 4 s and 8 s, each plus up to 1 s of jitter, between five
    attempts, and no wait after the last."""
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        await asyncio.wait_for(
            local_requests(raise_for_status=False, retry_max_attempts=5).get(base),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert len(served) == 5
    assert len(backoff_waits) == 4
    for retry, wait in enumerate(backoff_waits):
        assert 2**retry <= wait < 2**retry + 1
    # Jittered, so that clients throttled together don't retry in lockstep.
    assert any(wait != 2**retry for retry, wait in enumerate(backoff_waits))


@pytest.mark.asyncio
async def test_a_backoff_wait_never_passes_retry_max_wait(backoff_waits: list[float]):
    app, served = scripted(503)
    runner, base = await serve(app)

    try:
        await asyncio.wait_for(
            local_requests(
                raise_for_status=False, retry_max_attempts=6, retry_max_wait=1.5
            ).get(base),
            ENDS_WITHIN_SECONDS,
        )
    finally:
        await runner.cleanup()

    assert len(served) == 6
    assert max(backoff_waits) == 1.5
    assert all(wait <= 1.5 for wait in backoff_waits)


@pytest.mark.asyncio
async def test_a_budget_past_a_thousand_retries_waits_no_more_than_its_cap():
    """2 ** n is too large to add to a float from the 1,025th retry on; the
    backoff stops doubling well before that."""
    waits = await retry_waits(
        Requests(raise_for_status=False, retry_max_attempts=1100, retry_max_wait=0.01)
    )

    assert len(waits) == 1099
    assert max(waits) == 0.01


@pytest.mark.asyncio
async def test_a_call_cancelled_while_it_waits_to_retry_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
):
    """The wait between attempts is where a retrying call spends its time: a
    cancel there must end the call, not be taken for the wake-up that sends
    again."""
    waiting = asyncio.Event()

    async def hold_the_first_wait(seconds: float) -> None:
        if waiting.is_set():
            return
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(request_module, "sleep", hold_the_first_wait)
    app, served = scripted(503)
    runner, base = await serve(app)
    call = asyncio.create_task(local_requests(raise_for_status=False).get(base))

    try:
        await asyncio.wait_for(waiting.wait(), ENDS_WITHIN_SECONDS)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, ENDS_WITHIN_SECONDS)
    finally:
        call.cancel()
        await runner.cleanup()

    assert served == [503]


@pytest.mark.asyncio
async def test_a_redirect_loop_ends_at_max_redirects():
    app, received = _chain({"/a": [(302, "/a")]})
    runner, base = await serve(app)

    try:
        with pytest.raises(Exception, match="Too many redirects"):
            await asyncio.wait_for(
                local_requests(raise_for_status=False, retry_max_wait=0).get(
                    f"{base}/a", max_redirects=3
                ),
                ENDS_WITHIN_SECONDS,
            )
    finally:
        await runner.cleanup()

    assert len(received["/a"]) == 4
