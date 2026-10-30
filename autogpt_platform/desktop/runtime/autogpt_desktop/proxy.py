"""The public origin, standing in for single-container/nginx/nginx.conf.

Routing is the appliance's:

    /healthz          -> 200
    /_agpt/ws         -> websocket service /ws          (websocket)
    /_agpt/docs etc.  -> 404 (API docs and metrics stay private)
    /_agpt/<path>     -> REST service /<path>            (streamed, unbuffered)
    everything else   -> Next server

Responses are streamed chunk by chunk so AutoPilot's server-sent events reach
the browser as they are produced, and bodies are passed through still
compressed.

Cookies are kept apart from those of other installs of the app on the same
machine, which a browser would otherwise mix up (cookies.py).
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from aiohttp import web

from autogpt_desktop import cookies, install

logger = logging.getLogger("autogpt_desktop")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
PRIVATE_API = re.compile(
    r"^/_agpt/(?:(?:docs|redoc)(?:/|$)|(?:openapi\.json|metrics)/?$"
    r"|external-api/(?:(?:docs|redoc)(?:/|$)|(?:openapi\.json|metrics)/?$))"
)
API_REDIRECT = re.compile(r"^/(?:api|external-api)(?:/|$)")
MAX_BODY = 256 * 1024 * 1024

UPSTREAMS: web.AppKey[Upstreams] = web.AppKey("upstreams")
CLIENT: web.AppKey[aiohttp.ClientSession] = web.AppKey("client")
COOKIE_MARK: web.AppKey[str] = web.AppKey("cookie_mark")


@dataclass(frozen=True)
class Upstreams:
    public_url: str
    rest: str
    websocket: str
    frontend: str


def build_app(upstreams: Upstreams, install_name: str | None = None) -> web.Application:
    app = web.Application(client_max_size=MAX_BODY)
    app[UPSTREAMS] = upstreams
    app[COOKIE_MARK] = cookies.mark_of(install_name or install.name())
    app.cleanup_ctx.append(_client_session)
    app.router.add_route("*", "/{tail:.*}", _route)
    return app


async def _client_session(app: web.Application):
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=60)
    app[CLIENT] = aiohttp.ClientSession(connector=connector, auto_decompress=False)
    yield
    await app[CLIENT].close()


async def _route(request: web.Request) -> web.StreamResponse:
    upstreams = request.app[UPSTREAMS]
    path = request.path
    if path == "/healthz":
        return web.Response(text="ok\n")
    if path in ("/_agpt/ws", "/_agpt/ws/"):
        return await _websocket(request, f"{upstreams.websocket}/ws")
    if PRIVATE_API.match(path):
        raise web.HTTPNotFound()
    if path.startswith("/_agpt/"):
        target = upstreams.rest + request.rel_url.raw_path_qs[len("/_agpt") :]
        return await _http(request, target, read_timeout=3600, api=True)
    target = upstreams.frontend + request.rel_url.raw_path_qs
    if _is_websocket(request):
        return await _websocket(request, target.replace("http", "ws", 1))
    return await _http(request, target, read_timeout=300, api=False)


async def _http(
    request: web.Request, target: str, *, read_timeout: float, api: bool
) -> web.StreamResponse:
    upstreams = request.app[UPSTREAMS]
    client = request.app[CLIENT]
    try:
        async with client.request(
            request.method,
            target,
            headers=_forward_headers(request, upstreams),
            data=request.content if request.body_exists else None,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=read_timeout),
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status, reason=upstream.reason
            )
            for name, value in upstream.headers.items():
                if name.lower() in HOP_BY_HOP:
                    continue
                if name.lower() == "location":
                    value = _rewrite_location(value, upstreams, api)
                if name.lower() == "set-cookie":
                    value = cookies.to_browser(value, request.app[COOKIE_MARK])
                response.headers.add(name, value)
            if api:
                response.headers["X-Accel-Buffering"] = "no"
            if upstream.headers.get("Content-Length") and not _is_streaming(upstream):
                response.content_length = int(upstream.headers["Content-Length"])
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response
    except (TimeoutError, aiohttp.ClientConnectionError):
        raise web.HTTPBadGateway(
            text="AutoGPT is still starting or a service is down."
        ) from None


async def _websocket(request: web.Request, target: str) -> web.StreamResponse:
    upstreams = request.app[UPSTREAMS]
    client = request.app[CLIENT]
    protocols = [
        p.strip()
        for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        if p.strip()
    ]
    downstream = web.WebSocketResponse(
        protocols=protocols, max_msg_size=0, autoping=True
    )
    await downstream.prepare(request)
    query = request.rel_url.query_string
    url = f"{target}?{query}" if query and "?" not in target else target
    headers = {
        key: value
        for key, value in _forward_headers(request, upstreams).items()
        if not key.lower().startswith("sec-websocket")
    }
    try:
        async with client.ws_connect(
            url, headers=headers, protocols=protocols, max_msg_size=0, heartbeat=None
        ) as upstream:
            await asyncio.gather(
                _pump(downstream, upstream), _pump(upstream, downstream)
            )
    except aiohttp.ClientError:
        pass
    finally:
        await downstream.close()
    return downstream


async def _pump(source, sink) -> None:
    async for message in source:
        if message.type == aiohttp.WSMsgType.TEXT:
            await sink.send_str(message.data)
        elif message.type == aiohttp.WSMsgType.BINARY:
            await sink.send_bytes(message.data)
        else:
            break
    await sink.close()


def _forward_headers(request: web.Request, upstreams: Upstreams) -> dict[str, str]:
    public_host = upstreams.public_url.split("://", 1)[1]
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP and name.lower() != "cookie"
    }
    if cookie := _own_cookies(request):
        headers["Cookie"] = cookie
    headers["Host"] = public_host
    headers["X-Forwarded-For"] = request.remote or "127.0.0.1"
    headers["X-Forwarded-Proto"] = upstreams.public_url.split("://", 1)[0]
    headers["X-Forwarded-Host"] = public_host
    return headers


def _own_cookies(request: web.Request) -> str | None:
    """The cookies of this install among those the browser sent: it sends
    every install's, since they are all on 127.0.0.1 (cookies.py)."""
    sent = request.headers.getall("Cookie", [])
    if not sent:
        return None
    return cookies.to_platform("; ".join(sent), request.app[COOKIE_MARK])


def _rewrite_location(value: str, upstreams: Upstreams, api: bool) -> str:
    if api:
        try:
            location = urlsplit(value)
            rest = urlsplit(upstreams.rest)
            if (location.scheme or location.netloc) and (
                location.scheme != rest.scheme
                or location.hostname != rest.hostname
                or (location.port or (443 if location.scheme == "https" else 80))
                != (rest.port or (443 if rest.scheme == "https" else 80))
                or location.username is not None
                or location.password is not None
            ):
                return value
        except ValueError:
            return value
        if not API_REDIRECT.match(location.path):
            return value
        target = urlunsplit(("", "", location.path, location.query, location.fragment))
        return f"{upstreams.public_url}/_agpt{target}"
    # Next builds absolute redirects from its own listen address, and spells
    # the host either way, so both must be mapped back to the public origin.
    port = upstreams.frontend.rsplit(":", 1)[1]
    match = re.match(rf"^https?://(?:localhost|127\.0\.0\.1):{port}(/.*)?$", value)
    return upstreams.public_url + (match.group(1) or "/") if match else value


def _is_websocket(request: web.Request) -> bool:
    return request.headers.get("Upgrade", "").lower() == "websocket"


def _is_streaming(upstream: aiohttp.ClientResponse) -> bool:
    return upstream.headers.get("Content-Type", "").startswith("text/event-stream")


class ProxyThread:
    """Runs the proxy on its own event loop so the synchronous supervisor
    can keep its simple blocking style."""

    def __init__(self, upstreams: Upstreams, port: int) -> None:
        self.upstreams = upstreams
        self.port = port
        self.loop = asyncio.new_event_loop()
        self.runner: web.AppRunner | None = None
        self.started = threading.Event()
        self.failure: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="proxy", daemon=True)

    def start(self, timeout: float = 15) -> None:
        self.thread.start()
        if not self.started.wait(timeout):
            raise RuntimeError("the local web server did not start")
        if self.failure:
            raise RuntimeError(f"the local web server failed: {self.failure}")

    def stop(self) -> None:
        if self.runner:
            asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(
                10
            )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.set_exception_handler(_report_loop_error)
        try:
            # The window's websocket is still open when the app quits, and
            # aiohttp would wait a minute for it to finish on its own.
            self.runner = web.AppRunner(
                build_app(self.upstreams), access_log=None, shutdown_timeout=1
            )
            self.loop.run_until_complete(self.runner.setup())
            site = web.TCPSite(self.runner, "127.0.0.1", self.port)
            self.loop.run_until_complete(site.start())
        except BaseException as exc:
            self.failure = exc
            self.started.set()
            return
        self.started.set()
        self.loop.run_forever()


def _report_loop_error(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    if client_went_away(context):
        logger.debug(
            "a connection was dropped by its other end: %s", context.get("exception")
        )
        return
    loop.default_exception_handler(context)


def client_went_away(context: dict) -> bool:
    """Whether what the event loop reports is only a connection its other
    end dropped: a closed tab, a cancelled request, a service that let go of
    an idle connection. On Windows the loop finds out when it shuts its own
    end down, inside the transport's connection-lost callback, and reports it
    as an error with a traceback each time. Nothing was lost; anything else
    still goes to the default handler."""
    reset = isinstance(
        context.get("exception"), ConnectionResetError | ConnectionAbortedError
    )
    source = f"{context.get('message', '')} {context.get('handle', '')}"
    return reset and "_call_connection_lost" in source
