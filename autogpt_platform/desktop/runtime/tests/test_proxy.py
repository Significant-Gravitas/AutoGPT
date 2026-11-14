"""The proxy must behave like single-container/nginx/nginx.conf: same routes,
no buffering of streamed responses, websockets passed through."""

import asyncio
import socket

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from autogpt_desktop.proxy import (
    ProxyThread,
    Upstreams,
    _report_loop_error,
    _rewrite_location,
    client_went_away,
)

RELEASE: web.AppKey[asyncio.Event] = web.AppKey("release")
REQUESTS: web.AppKey[list[str]] = web.AppKey("requests")


@pytest.mark.parametrize(
    "location, expected",
    [
        ("/api/callback?x=1#done", "http://127.0.0.1:4000/_agpt/api/callback?x=1#done"),
        ("/external-api?x=1", "http://127.0.0.1:4000/_agpt/external-api?x=1"),
        (
            "http://127.0.0.1:4001/api/callback",
            "http://127.0.0.1:4000/_agpt/api/callback",
        ),
        (
            "https://provider.example/api/authorize?x=1",
            "https://provider.example/api/authorize?x=1",
        ),
        ("http://127.0.0.1:9999/api/callback", "http://127.0.0.1:9999/api/callback"),
        ("https://127.0.0.1:4001/api/callback", "https://127.0.0.1:4001/api/callback"),
        (
            "http://user@127.0.0.1:4001/api/callback",
            "http://user@127.0.0.1:4001/api/callback",
        ),
        ("//provider.example/api/callback", "//provider.example/api/callback"),
        ("http://[invalid/api/callback", "http://[invalid/api/callback"),
        ("https://127.0.0.1:4000/api/callback", "https://127.0.0.1:4000/api/callback"),
        (
            "http://127.0.0.1.example:4000/api/callback",
            "http://127.0.0.1.example:4000/api/callback",
        ),
        (
            "http://user@127.0.0.1:4000/api/callback",
            "http://user@127.0.0.1:4000/api/callback",
        ),
        ("http://127.0.0.1:4000/login", "http://127.0.0.1:4000/login"),
        (
            "http://127.0.0.1:4000/_agpt/api/callback",
            "http://127.0.0.1:4000/_agpt/api/callback",
        ),
        ("http://127.0.0.1:bad/api/callback", "http://127.0.0.1:bad/api/callback"),
        ("/apiculture", "/apiculture"),
        ("relative/api/callback", "relative/api/callback"),
    ],
)
def test_api_redirects_preserve_external_origins(location, expected):
    upstreams = Upstreams("http://127.0.0.1:4000", "http://127.0.0.1:4001", "", "")
    assert _rewrite_location(location, upstreams, api=True) == expected


@pytest.mark.parametrize("prefix", ["api", "external-api"])
@pytest.mark.parametrize(
    "public, origin",
    [
        ("http://127.0.0.1:4000", "http://127.0.0.1:4000"),
        ("http://127.0.0.1:80", "http://127.0.0.1"),
        ("https://127.0.0.1", "https://127.0.0.1:443"),
        ("http://[::1]:4000", "http://[::1]:4000"),
    ],
)
def test_api_redirects_from_public_origin_keep_prefix(public, origin, prefix):
    upstreams = Upstreams(public, "http://127.0.0.1:4001", "", "")
    path = f"/{prefix}/callback/?next=%2Fhome#done"
    assert _rewrite_location(f"{origin}{path}", upstreams, api=True) == (
        f"{public}/_agpt{path}"
    )


def test_api_redirect_does_not_treat_explicit_zero_port_as_default():
    upstreams = Upstreams("http://127.0.0.1", "http://127.0.0.1:4001", "", "")
    location = "http://127.0.0.1:0/api/callback"
    assert _rewrite_location(location, upstreams, api=True) == location


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def start(app: web.Application) -> tuple[web.AppRunner, int]:
    port = free_port()
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, port


def rest_app() -> web.Application:
    async def echo(request: web.Request) -> web.Response:
        request.app[REQUESTS].append(request.raw_path)
        return web.json_response(
            {
                "path": request.path_qs,
                "host": request.headers["Host"],
                "forwarded_host": request.headers.get("X-Forwarded-Host"),
                "forwarded_for": request.headers.get("X-Forwarded-For"),
                "forwarded_proto": request.headers.get("X-Forwarded-Proto"),
                "body": (await request.read()).decode(),
            }
        )

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/api/integrations/callback?x=1")

    async def slash_redirect(request: web.Request) -> web.Response:
        raise web.HTTPTemporaryRedirect(
            str(request.url.with_path(request.path + "/", keep_query=True))
        )

    async def stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        await request.app[RELEASE].wait()
        await response.write(b"data: second\n\n")
        return response

    async def partial(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        if request.query["framing"] == "length":
            response.content_length = 10
        await response.prepare(request)
        await response.write(b"first")
        await request.app[RELEASE].wait()
        return response

    async def cookies(request: web.Request) -> web.Response:
        response = web.Response(text="ok")
        response.headers.add("Set-Cookie", "a=1; Path=/")
        response.headers.add("Set-Cookie", "b=2; Path=/")
        return response

    app = web.Application()
    app[RELEASE] = asyncio.Event()
    app[REQUESTS] = []
    app.router.add_get("/api/redirect", redirect)
    app.router.add_get("/api/normalize", slash_redirect)
    app.router.add_get("/external-api/normalize", slash_redirect)
    app.router.add_get("/api/stream", stream)
    app.router.add_get("/api/partial", partial)
    app.router.add_get("/api/cookies", cookies)
    app.router.add_route("*", "/{tail:.*}", echo)
    return app


def websocket_app() -> web.Application:
    async def handler(request: web.Request) -> web.WebSocketResponse:
        socket_ = web.WebSocketResponse()
        await socket_.prepare(request)
        await socket_.send_str(f"hello {request.query.get('token')}")
        async for message in socket_:
            await socket_.send_str(message.data.upper())
        return socket_

    app = web.Application()
    app.router.add_get("/ws", handler)
    return app


def frontend_app() -> web.Application:
    async def page(request: web.Request) -> web.Response:
        return web.Response(text=f"frontend {request.path}")

    async def to_login(request: web.Request) -> web.Response:
        # Next builds this from its own listen address, as `localhost`.
        port = request.transport.get_extra_info("sockname")[1]
        location = f"http://localhost:{port}/login?next=%2Fhome"
        return web.Response(status=307, headers={"Location": location})

    app = web.Application()
    app.router.add_get("/home", to_login)
    app.router.add_route("*", "/{tail:.*}", page)
    return app


@pytest.fixture
async def stack():
    rest = rest_app()
    runners = []
    ports = []
    for app in (rest, websocket_app(), frontend_app()):
        runner, port = await start(app)
        runners.append(runner)
        ports.append(port)
    public = free_port()
    proxy = ProxyThread(
        Upstreams(
            public_url=f"http://127.0.0.1:{public}",
            rest=f"http://127.0.0.1:{ports[0]}",
            websocket=f"http://127.0.0.1:{ports[1]}",
            frontend=f"http://127.0.0.1:{ports[2]}",
        ),
        public,
    )
    await asyncio.to_thread(proxy.start)
    async with aiohttp.ClientSession() as client:
        yield client, f"http://127.0.0.1:{public}", rest
    await asyncio.to_thread(proxy.stop)
    for runner in runners:
        await runner.cleanup()


async def test_health_endpoint_is_answered_by_the_proxy(stack):
    client, base, _ = stack
    async with client.get(f"{base}/healthz") as response:
        assert response.status == 200
        assert await response.text() == "ok\n"


async def test_api_prefix_is_stripped_and_public_host_is_forwarded(stack):
    client, base, _ = stack
    async with client.post(
        f"{base}/_agpt/api/graphs?page=2", data=b"payload"
    ) as response:
        body = await response.json()
    assert body["path"] == "/api/graphs?page=2"
    assert body["body"] == "payload"
    assert body["host"] == body["forwarded_host"] == base.split("://")[1]
    assert response.headers["X-Accel-Buffering"] == "no"


@pytest.mark.parametrize(
    "path",
    [
        "/_agpt/docs",
        "/_agpt/redoc",
        "/_agpt/openapi.json",
        "/_agpt/metrics",
        "/_agpt/external-api/docs",
    ],
)
async def test_api_documentation_and_metrics_stay_private(stack, path):
    client, base, _ = stack
    async with client.get(f"{base}{path}") as response:
        assert response.status == 404


@pytest.mark.parametrize(
    "endpoint",
    [
        "docs",
        "redoc",
        "openapi.json",
        "metrics",
        "external-api/docs",
        "external-api/redoc",
        "external-api/openapi.json",
        "external-api/metrics",
    ],
)
@pytest.mark.parametrize(
    "prefix", ["x/../", "./", "x/%2e%2e/", "%2E/", "x/.%2E/", "x%2F..%2F"]
)
async def test_dot_segments_cannot_reach_private_api_endpoints(stack, prefix, endpoint):
    client, base, rest = stack
    target = URL(f"{base}/_agpt/{prefix}{endpoint}?probe=1", encoded=True)
    async with client.get(target) as response:
        body = await response.text()
        assert response.status == 404, body
    assert rest[REQUESTS] == []


@pytest.mark.parametrize(
    "path",
    [
        "/api/files/report..txt",
        "/api/files/.config",
        "/api/files/...",
        "/api/echo?next=/x/../docs",
    ],
)
async def test_dots_in_names_and_queries_remain_public(stack, path):
    client, base, _ = stack
    async with client.get(URL(f"{base}/_agpt{path}", encoded=True)) as response:
        assert response.status == 200
        assert (await response.json())["path"] == path


async def test_everything_else_goes_to_the_frontend(stack):
    client, base, _ = stack
    async with client.get(f"{base}/library/agents") as response:
        assert await response.text() == "frontend /library/agents"


async def test_forwarded_identity_cannot_be_spoofed_by_a_client(stack):
    client, base, _ = stack
    async with client.get(
        f"{base}/_agpt/api/headers",
        headers={
            "X-Forwarded-For": "203.0.113.9",
            "X-Forwarded-Host": "untrusted.example",
            "X-Forwarded-Proto": "https",
        },
    ) as response:
        body = await response.json()
    assert body["forwarded_for"] == "127.0.0.1"
    assert body["forwarded_host"] == base.split("://")[1]
    assert body["forwarded_proto"] == "http"


async def test_frontend_redirects_never_leak_the_internal_address(stack):
    client, base, _ = stack
    async with client.get(f"{base}/home", allow_redirects=False) as response:
        assert response.status == 307
        assert response.headers["Location"] == f"{base}/login?next=%2Fhome"


async def test_api_redirects_are_rewritten_to_the_public_origin(stack):
    client, base, _ = stack
    async with client.get(
        f"{base}/_agpt/api/redirect", allow_redirects=False
    ) as response:
        assert response.status == 302
        assert (
            response.headers["Location"]
            == f"{base}/_agpt/api/integrations/callback?x=1"
        )


@pytest.mark.parametrize("prefix", ["api", "external-api"])
async def test_public_host_slash_redirect_returns_to_rest(stack, prefix):
    client, base, rest = stack
    async with client.get(f"{base}/_agpt/{prefix}/normalize?x=1") as response:
        assert response.status == 200
        assert len(response.history) == 1
        assert response.history[0].status == 307
        assert response.history[0].headers["Location"] == (
            f"{base}/_agpt/{prefix}/normalize/?x=1"
        )
        assert (await response.json())["path"] == f"/{prefix}/normalize/?x=1"
    assert rest[REQUESTS] == [f"/{prefix}/normalize/?x=1"]


async def test_every_set_cookie_header_survives(stack):
    client, base, _ = stack
    async with client.get(f"{base}/_agpt/api/cookies") as response:
        assert response.headers.getall("Set-Cookie") == ["a=1; Path=/", "b=2; Path=/"]


async def test_event_streams_are_delivered_as_they_are_produced(stack):
    client, base, rest = stack
    async with client.get(f"{base}/_agpt/api/stream") as response:
        first = await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=5)
        assert first == b"data: first\n\n"
        rest[RELEASE].set()
        second = await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=5)
        assert second == b"data: second\n\n"


@pytest.mark.parametrize(
    "error", [TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError]
)
@pytest.mark.parametrize("framing", ["length", "chunked"])
@pytest.mark.parametrize("failure_point", ["prepare", "before-body", "after-body"])
async def test_stream_failures_truncate_without_second_response(
    stack, monkeypatch, error, framing, failure_point
):
    _, base, rest = stack
    original = aiohttp.StreamReader.iter_any

    async def interrupted(reader):
        if failure_point == "after-body":
            async for chunk in original(reader):
                yield chunk
                break
        raise error("injected upstream stream failure")

    if failure_point == "prepare":
        original_prepare = web.StreamResponse.prepare

        async def interrupted_prepare(response, request):
            result = await original_prepare(response, request)
            if request.path.startswith("/_agpt/") and response.status == 200:
                raise error("injected failure after sending headers")
            return result

        monkeypatch.setattr(web.StreamResponse, "prepare", interrupted_prepare)
    else:
        monkeypatch.setattr(aiohttp.StreamReader, "iter_any", interrupted)
    reader, writer = await asyncio.open_connection("127.0.0.1", URL(base).port)
    try:
        writer.write(
            (
                f"GET /_agpt/api/partial?framing={framing} HTTP/1.1\r\n"
                f"Host: {URL(base).authority}\r\nConnection: close\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=30)
        assert raw.startswith(b"HTTP/1.1 200 OK\r\n"), raw
        assert raw.count(b"HTTP/1.1 ") == 1, raw
        headers, body = raw.split(b"\r\n\r\n", 1)
        if framing == "length":
            assert b"Content-Length: 10\r\n" in headers
            assert body == (b"first" if failure_point == "after-body" else b"")
        else:
            assert b"Transfer-Encoding: chunked\r\n" in headers
            assert body == (b"5\r\nfirst\r\n" if failure_point == "after-body" else b"")
    finally:
        rest[RELEASE].set()
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize("error", [TimeoutError, aiohttp.ClientConnectionError])
async def test_failures_before_upstream_headers_remain_bad_gateway(
    stack, monkeypatch, error
):
    client, base, _ = stack
    original = aiohttp.ClientSession._request

    async def unavailable(session, method, url, **kwargs):
        if URL(url).path == "/api/fail-before-headers":
            raise error("injected upstream request failure")
        return await original(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unavailable)
    async with client.get(f"{base}/_agpt/api/fail-before-headers") as response:
        assert response.status == 502
        assert (
            await response.text() == "AutoGPT is still starting or a service is down."
        )


async def test_websockets_are_proxied_with_their_query_string(stack):
    client, base, _ = stack
    async with client.ws_connect(f"{base}/_agpt/ws?token=abc") as socket_:
        assert await socket_.receive_str() == "hello abc"
        await socket_.send_str("ping")
        assert await socket_.receive_str() == "PING"


async def test_a_down_upstream_is_a_bad_gateway_not_a_crash(stack):
    client, base, _ = stack
    proxy_with_dead_upstream = ProxyThread(
        Upstreams(
            public_url="http://127.0.0.1:1",
            rest=f"http://127.0.0.1:{free_port()}",
            websocket=f"http://127.0.0.1:{free_port()}",
            frontend=f"http://127.0.0.1:{free_port()}",
        ),
        port := free_port(),
    )
    await asyncio.to_thread(proxy_with_dead_upstream.start)
    try:
        async with client.get(f"http://127.0.0.1:{port}/_agpt/api/x") as response:
            assert response.status == 502
    finally:
        await asyncio.to_thread(proxy_with_dead_upstream.stop)


# --- what the proxy's event loop reports ---------------------------------------


class _ProactorBasePipeTransport:
    """Named and shaped like asyncio's, whose callback is what fails on
    Windows when a browser drops a connection."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def _call_connection_lost(self, exc) -> None:
        raise self.error

    def _loop_reading(self, future) -> None:
        raise self.error


def reported_by_a_loop(callback, *arguments) -> dict:
    """What asyncio itself hands an exception handler when `callback` fails."""
    loop = asyncio.new_event_loop()
    seen: list[dict] = []
    loop.set_exception_handler(lambda loop, context: seen.append(context))
    try:
        loop.call_soon(callback, *arguments)
        loop.call_soon(loop.stop)
        loop.run_forever()
    finally:
        loop.close()
    (context,) = seen
    return context


class RecordingLoop(asyncio.BaseEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self.passed_on: list[dict] = []

    def default_exception_handler(self, context: dict) -> None:
        self.passed_on.append(context)


@pytest.mark.parametrize(
    "error", [ConnectionResetError(10054, "reset"), ConnectionAbortedError()]
)
def test_a_connection_dropped_by_its_other_end_is_logged_quietly(error, caplog):
    transport = _ProactorBasePipeTransport(error)
    context = reported_by_a_loop(transport._call_connection_lost, None)
    # As in a runtime.log from Windows: "Exception in callback
    # _ProactorBasePipeTransport._call_connection_lost()".
    assert "_ProactorBasePipeTransport._call_connection_lost" in context["message"]
    assert client_went_away(context)

    loop = RecordingLoop()
    with caplog.at_level("DEBUG", logger="autogpt_desktop"):
        _report_loop_error(loop, context)
    loop.close()

    assert loop.passed_on == []
    (record,) = [
        r for r in caplog.records if "dropped by its other end" in r.getMessage()
    ]
    assert record.levelname == "DEBUG" and not record.exc_info


@pytest.mark.parametrize(
    ("error", "callback"),
    [
        (RuntimeError("a bug"), "_call_connection_lost"),  # another error, same place
        (
            ConnectionResetError(10054, "reset"),
            "_loop_reading",
        ),  # same error, elsewhere
        (OSError("disk"), "_loop_reading"),
    ],
)
def test_anything_else_the_loop_reports_still_reaches_the_default_handler(
    error, callback
):
    transport = _ProactorBasePipeTransport(error)
    context = reported_by_a_loop(getattr(transport, callback), None)
    assert not client_went_away(context)

    loop = RecordingLoop()
    _report_loop_error(loop, context)
    loop.close()

    assert loop.passed_on == [context]


def test_a_report_without_an_exception_is_not_mistaken_for_one():
    assert not client_went_away({"message": "Task was destroyed but it is pending!"})
    assert not client_went_away({})
