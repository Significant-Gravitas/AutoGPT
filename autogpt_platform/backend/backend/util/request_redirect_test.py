"""How ``Requests`` follows a redirect: the method and body each hop is sent
with, as RFC 9110 §15.4 has it (#17145).

Every test drives real local servers and asserts what each hop received."""

import asyncio
import io
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import ParseResult

import pytest
from aiohttp import web
from pydantic import BaseModel

from backend.util.request import HTTPClientError, Requests
from backend.util.request_test_helpers import ENDS_WITHIN_SECONDS, local_requests, serve


class Received(BaseModel):
    """One request as a test server received it, with its header names
    lower-cased. A multipart body's boundary, new on every send, reads
    ``BOUNDARY`` in the body and the Content-Type alike."""

    server: str
    method: str
    path: str
    content_type: str | None
    body: bytes
    headers: dict[str, str]


Answer = Callable[[web.Request], web.StreamResponse]


class Origin:
    """A local server answering ``routes`` by path, which notes each request
    it receives in a list it shares with the other origins of a test."""

    def __init__(self, name: str, received: list[Received]):
        self.name = name
        self.routes: dict[str, Answer] = {}
        self.base = ""
        self._received = received

    def app(self) -> web.Application:
        async def handle(request: web.Request) -> web.StreamResponse:
            self._received.append(_received(self.name, request, await request.read()))
            answer = self.routes.get(request.path)
            return answer(request) if answer else web.Response(status=404)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        return app


@asynccontextmanager
async def origins() -> AsyncIterator[tuple[Origin, Origin, list[Received]]]:
    """Two local servers on two ports, so two origins, and what they receive."""
    received: list[Received] = []
    first, second = Origin("first", received), Origin("second", received)
    first_runner, first.base = await serve(first.app())
    second_runner, second.base = await serve(second.app())
    try:
        yield first, second, received
    finally:
        await first_runner.cleanup()
        await second_runner.cleanup()


def _received(server: str, request: web.Request, body: bytes) -> Received:
    content_type = request.headers.get("Content-Type")
    boundary = re.search(r"boundary=([^;\s]+)", content_type or "")
    if content_type is not None and boundary:
        content_type = content_type.replace(boundary.group(1), "BOUNDARY")
        body = body.replace(boundary.group(1).encode(), b"BOUNDARY")
    return Received(
        server=server,
        method=request.method,
        path=request.path,
        content_type=content_type,
        body=body,
        headers={name.lower(): value for name, value in request.headers.items()},
    )


def redirect(status: int, location: str) -> Answer:
    return lambda request: web.Response(status=status, headers={"Location": location})


def done(request: web.Request) -> web.Response:
    return web.Response(text="done")


def answers(*steps: Answer) -> Answer:
    """Each step in turn, then the last one for good."""
    sent: list[None] = []

    def answer(request: web.Request) -> web.StreamResponse:
        sent.append(None)
        return steps[min(len(sent), len(steps)) - 1](request)

    return answer


def status(code: int) -> Answer:
    return lambda request: web.Response(status=code, text=f"status {code}")


def location(target: Origin, origin: str) -> str:
    """Where a redirect from the first origin sends the next hop."""
    return "/next" if origin == "same-origin" else f"{target.base}/next"


async def bounded(sent):
    return await asyncio.wait_for(sent, ENDS_WITHIN_SECONDS)


def hops(received: list[Received]) -> list[tuple[str, str, str]]:
    return [(hop.server, hop.method, hop.path) for hop in received]


def without_body(hop: Received) -> bool:
    return (
        hop.body == b""
        and hop.content_type is None
        and "content-length" not in hop.headers
        and "transfer-encoding" not in hop.headers
    )


# --- A POST, with each kind of body, through each redirect ---

# Each kind of POST body: the kwargs that send it, then the Content-Type and
# the part of the body its first hop must receive.
POST_BODIES: dict[str, Callable[[], tuple[dict[str, Any], str, bytes]]] = {
    "data": lambda: (
        {"data": {"order": "1", "note": "ok"}},
        "application/x-www-form-urlencoded",
        b"order=1&note=ok",
    ),
    "json": lambda: ({"json": {"order": 1}}, "application/json", b'{"order": 1}'),
    "files": lambda: (
        {
            "data": {"note": "ok"},
            "files": [
                ("sheet", ("order.csv", io.BytesIO(b"id,qty\n1,2\n"), "text/csv"))
            ],
        },
        "multipart/form-data; boundary=BOUNDARY",
        b'name="sheet"; filename="order.csv"\r\n\r\nid,qty\n1,2\n',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["same-origin", "cross-origin"])
@pytest.mark.parametrize("body", ["data", "json", "files"])
@pytest.mark.parametrize(
    "status, keeps_the_post",
    [(301, False), (302, False), (303, False), (307, True), (308, True)],
)
async def test_a_redirected_post_goes_on_as_rfc_9110_has_it(
    status: int, keeps_the_post: bool, body: str, origin: str
):
    """A 301, 302 or 303 makes the next hop a GET without the body or its
    Content-Type; a 307 or 308 sends the same POST, body and all."""
    kwargs, content_type, sent = POST_BODIES[body]()
    async with origins() as (first, second, received):
        first.routes["/start"] = redirect(status, location(second, origin))
        first.routes["/next"] = second.routes["/next"] = done
        response = await bounded(local_requests().post(f"{first.base}/start", **kwargs))

    assert response.status == 200
    landing = "first" if origin == "same-origin" else "second"
    assert hops(received) == [
        ("first", "POST", "/start"),
        (landing, "POST" if keeps_the_post else "GET", "/next"),
    ]
    initial, following = received
    # The body did go out, so the next hop can show it was left behind.
    assert initial.content_type == content_type
    assert sent in initial.body
    if keeps_the_post:
        assert following.content_type == initial.content_type
        assert following.body == initial.body
    else:
        assert without_body(following)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, status, next_method, keeps_body",
    [
        # A 303 makes the next hop a GET without the body, whatever the method.
        *[(m, 303, "GET", False) for m in ("GET", "PUT", "PATCH", "DELETE", "OPTIONS")],
        # A 301 or 302 changes only a POST, whatever the case it is named in.
        *[
            (m, s, m, True)
            for m in ("GET", "PUT", "PATCH", "DELETE", "OPTIONS")
            for s in (301, 302)
        ],
        ("post", 301, "GET", False),
        ("Post", 302, "GET", False),
        # A 307 or 308 never changes the method.
        *[
            (m, s, m, True)
            for m in ("GET", "PUT", "PATCH", "DELETE", "OPTIONS")
            for s in (307, 308)
        ],
    ],
)
async def test_each_method_goes_on_as_rfc_9110_has_it(
    method: str, status: int, next_method: str, keeps_body: bool
):
    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(status, "/next")
        first.routes["/next"] = done
        response = await bounded(
            local_requests().request(method, f"{first.base}/start", json={"order": 1})
        )

    assert response.status == 200
    assert hops(received) == [
        ("first", method.upper(), "/start"),
        ("first", next_method, "/next"),
    ]
    initial, following = received
    assert initial.body == b'{"order": 1}'
    if keeps_body:
        assert (following.body, following.content_type) == (
            initial.body,
            "application/json",
        )
    else:
        assert without_body(following)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_a_head_stays_a_head(status: int):
    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(status, "/next")
        first.routes["/next"] = done
        response = await bounded(local_requests().head(f"{first.base}/start"))

    assert response.status == 200
    assert hops(received) == [("first", "HEAD", "/start"), ("first", "HEAD", "/next")]


# --- What goes with the body ---

# The headers a caller may set that describe the body.
BODY_HEADERS = {
    "Content-Type": "application/vnd.order+json",
    "Content-Language": "en-GB",
    "Content-Location": "/orders/new",
    "Digest": "sha-256=X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=",
    "Repr-Digest": "sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:",
    "Trailer": "X-Order-Checksum",
    "Expect": "100-continue",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["same-origin", "cross-origin"])
@pytest.mark.parametrize(
    "where", ["headers", "extra_headers"], ids=["call-headers", "client-headers"]
)
@pytest.mark.parametrize("status, keeps_the_post", [(303, False), (307, True)])
async def test_the_headers_that_describe_a_body_go_with_it(
    status: int, keeps_the_post: bool, where: str, origin: str
):
    """``extra_headers`` is merged into every hop again, so a header it holds
    has to stay dropped for the rest of the chain, not only for one hop."""
    ordinary = {"X-Request-Id": "order-1"}
    client = (
        local_requests(extra_headers=BODY_HEADERS)
        if where == "extra_headers"
        else local_requests()
    )
    call_headers = {**ordinary, **(BODY_HEADERS if where == "headers" else {})}
    async with origins() as (first, second, received):
        first.routes["/start"] = redirect(status, location(second, origin))
        first.routes["/next"] = second.routes["/next"] = done
        await bounded(
            client.post(f"{first.base}/start", json={"order": 1}, headers=call_headers)
        )

    initial, following = received
    described = {name.lower(): value for name, value in BODY_HEADERS.items()}
    assert described.items() <= initial.headers.items()
    assert following.headers["x-request-id"] == "order-1"
    if keeps_the_post:
        assert described.items() <= following.headers.items()
    else:
        assert not described.keys() & following.headers.keys()
        assert without_body(following)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option, header",
    [({"chunked": True}, "transfer-encoding"), ({"expect100": True}, "expect")],
    ids=["chunked", "expect100"],
)
@pytest.mark.parametrize("status, keeps_the_post", [(303, False), (307, True)])
async def test_the_options_that_frame_a_body_go_with_it(
    status: int, keeps_the_post: bool, option: dict, header: str
):
    """``chunked`` would end a GET with a chunked body's terminator, which the
    server reads as a second request and refuses with a 400."""
    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(status, "/next")
        first.routes["/next"] = done
        response = await bounded(
            local_requests().post(f"{first.base}/start", json={"order": 1}, **option)
        )

    assert response.status == 200
    assert hops(received) == [
        ("first", "POST", "/start"),
        ("first", "POST" if keeps_the_post else "GET", "/next"),
    ]
    initial, following = received
    assert header in initial.headers
    assert (header in following.headers) is keeps_the_post


class CountingBytesIO(io.BytesIO):
    """A BytesIO that counts every call that reads it or moves its position."""

    def __init__(self, content: bytes):
        super().__init__(content)
        self.touches = 0

    def read(self, size: int | None = -1) -> bytes:
        self.touches += 1
        return super().read(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.touches += 1
        return super().seek(offset, whence)

    def tell(self) -> int:
        self.touches += 1
        return super().tell()

    def getbuffer(self) -> memoryview:
        self.touches += 1
        return super().getbuffer()


def _counted_body(kind: str, stream: CountingBytesIO) -> dict[str, Any]:
    if kind == "upload":
        return {"files": [("sheet", ("order.csv", stream, "text/csv"))]}
    return {"data": stream}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["upload", "body"])
async def test_a_body_left_behind_is_not_read_again(kind: str):
    """The body a 303 leaves behind is neither rewound, nor read, nor made
    into a form for the next hop: the stream is touched exactly as often as
    by one send that is never redirected, while a 307 does touch it again."""
    touches: dict[int, int] = {}
    for status in (200, 303, 307):
        stream = CountingBytesIO(b"id,qty\n1,2\n")
        async with origins() as (first, _, received):
            first.routes["/start"] = (
                done if status == 200 else redirect(status, "/next")
            )
            first.routes["/next"] = done
            response = await bounded(
                local_requests().post(
                    f"{first.base}/start", **_counted_body(kind, stream)
                )
            )
        assert response.status == 200
        assert len(received) == (1 if status == 200 else 2)
        touches[status] = stream.touches

    assert touches[200] > 0
    assert touches[303] == touches[200]
    assert touches[307] > touches[200]


@pytest.mark.asyncio
@pytest.mark.parametrize("then", [301, 302, 307, 308])
async def test_a_body_left_behind_stays_behind_for_the_rest_of_the_chain(then: int):
    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(303, "/middle")
        first.routes["/middle"] = redirect(then, "/next")
        first.routes["/next"] = done
        response = await bounded(
            local_requests(extra_headers={"Content-Type": "application/json"}).post(
                f"{first.base}/start", json={"order": 1}
            )
        )

    assert response.status == 200
    assert hops(received) == [
        ("first", "POST", "/start"),
        ("first", "GET", "/middle"),
        ("first", "GET", "/next"),
    ]
    assert all(without_body(hop) for hop in received[1:])


# --- Each hop is still checked and retried as before ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_kwargs, first_answers, next_answers, outcome, expected",
    [
        pytest.param(
            {"raise_for_status": False},
            [redirect(303, "/next")],
            [status(503), done],
            200,
            [("POST", "/start"), ("GET", "/next"), ("GET", "/next")],
            id="the-get-is-sent-again-not-the-post",
        ),
        pytest.param(
            {"raise_for_status": False, "retry_max_attempts": 3},
            [status(503), redirect(303, "/next")],
            [status(503)],
            503,
            [
                ("POST", "/start"),
                ("POST", "/start"),
                ("GET", "/next"),
                ("GET", "/next"),
            ],
            id="one-budget-for-the-post-and-the-get",
        ),
        pytest.param(
            {"retry_max_attempts": 3},
            [redirect(302, "/next")],
            [status(429)],
            429,
            [("POST", "/start"), ("GET", "/next"), ("GET", "/next"), ("GET", "/next")],
            id="raising-once-the-gets-run-out",
        ),
    ],
)
async def test_a_hop_turned_into_a_get_is_retried_within_the_one_budget(
    client_kwargs: dict,
    first_answers: list[Answer],
    next_answers: list[Answer],
    outcome: int,
    expected: list[tuple[str, str]],
):
    async with origins() as (first, _, received):
        first.routes["/start"] = answers(*first_answers)
        first.routes["/next"] = answers(*next_answers)
        sent = local_requests(retry_max_wait=0, **client_kwargs).post(
            f"{first.base}/start", json={"order": 1}
        )
        if client_kwargs.get("raise_for_status", True):
            with pytest.raises(HTTPClientError) as raised:
                await bounded(sent)
            assert raised.value.status_code == outcome
        else:
            assert (await bounded(sent)).status == outcome

    assert [(hop.method, hop.path) for hop in received] == expected
    assert all(hop.body == b'{"order": 1}' for hop in received if hop.method == "POST")
    assert all(without_body(hop) for hop in received if hop.method == "GET")


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_status", [301, 302, 303])
@pytest.mark.parametrize(
    "first_sends", [1, 2], ids=["redirects", "throttled-then-redirects"]
)
async def test_a_get_to_an_address_the_client_does_not_trust_is_never_sent(
    redirect_status: int, first_sends: int
):
    """The hop a redirect turned into a GET is checked against the blocked
    ranges like any other: the second origin is loopback, which this client
    does not trust, so nothing reaches it."""
    async with origins() as (first, second, received):
        first.routes["/start"] = answers(
            *[status(503)] * (first_sends - 1),
            redirect(redirect_status, f"{second.base}/next"),
        )
        second.routes["/next"] = done
        client = Requests(
            trusted_origins=[first.base.removeprefix("http://")],
            raise_for_status=False,
            retry_max_attempts=3,
            retry_max_wait=0,
        )
        with pytest.raises(ValueError, match="blocked or private IP"):
            await bounded(client.post(f"{first.base}/start", json={"order": 1}))

    assert hops(received) == [("first", "POST", "/start")] * first_sends


@pytest.mark.asyncio
async def test_every_send_of_a_hop_turned_into_a_get_is_checked():
    checked: list[str] = []

    def note(url: ParseResult) -> ParseResult:
        checked.append(url.path)
        return url

    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(303, "/next")
        first.routes["/next"] = answers(status(503), status(503), done)
        response = await bounded(
            local_requests(
                raise_for_status=False,
                retry_max_attempts=4,
                retry_max_wait=0,
                extra_url_validator=note,
            ).post(f"{first.base}/start", json={"order": 1})
        )

    assert response.status == 200
    assert checked == ["/start", "/next", "/next", "/next"]
    assert [hop.method for hop in received] == ["POST", "GET", "GET", "GET"]


@pytest.mark.asyncio
async def test_a_hop_turned_into_a_get_counts_towards_max_redirects():
    async with origins() as (first, _, received):
        first.routes["/start"] = redirect(303, "/start")
        with pytest.raises(Exception, match="Too many redirects"):
            await bounded(
                local_requests().post(
                    f"{first.base}/start", json={"order": 1}, max_redirects=2
                )
            )

    assert [hop.method for hop in received] == ["POST", "GET", "GET"]
