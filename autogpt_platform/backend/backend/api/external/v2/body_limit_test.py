"""A request body is bounded before the app holds it, whoever reads it.

FastAPI and FastMCP both read a body whole before acting on it, so the bound
has to sit in front of them, and its 413 has to survive whatever each does
with a read that fails.
"""

import contextlib
import json

import fastapi
import pytest
import pytest_mock
from fastapi import params
from fastapi.testclient import TestClient
from starlette.formparsers import FormParser
from starlette.responses import JSONResponse

from backend.api.external.v2 import body_limit, global_rate_limit
from backend.api.external.v2.app import v2_app
from backend.api.external.v2.body_limit import (
    MAX_BODY_BYTES,
    BodySizeLimitMiddleware,
    max_body_bytes,
)
from backend.api.middleware.body_limit import MULTIPART_REQUIRED

MiB = 1024 * 1024
MULTIPART = "multipart/form-data; boundary=x"
URLENCODED = "application/x-www-form-urlencoded"


async def test_a_body_declared_over_the_cap_is_refused_before_the_app_runs() -> None:
    called = []

    async def app(scope, receive, send):
        called.append(scope)

    sent = await _call(
        BodySizeLimitMiddleware(app),
        _scope("/mcp", headers=[(b"content-length", str(MAX_BODY_BYTES + 1).encode())]),
        _chunks(b""),
    )

    assert not called
    assert _statuses(sent) == [413]
    error = _body(sent)["error"]
    assert error["code"] == "payload_too_large"
    assert error["details"] == {"limit_bytes": MAX_BODY_BYTES}


async def test_a_body_sent_in_chunks_is_refused_when_it_crosses_the_cap() -> None:
    """No length to check up front: FastAPI would answer the failed read 400."""
    reached = []
    api = fastapi.FastAPI()

    @api.post("/graphs")
    async def create(body: dict) -> dict:
        reached.append(body)
        return {}

    piece = b" " * MiB
    sent = await _call(
        BodySizeLimitMiddleware(api),
        _scope("/graphs"),
        _chunks(b"{", *[piece] * 11, b"}"),
    )

    assert not reached
    assert _statuses(sent) == [413]
    assert _body(sent)["error"]["code"] == "payload_too_large"


async def test_an_app_that_answers_a_failed_read_itself_cannot_replace_the_413() -> (
    None
):
    """FastMCP catches the read error and answers 500; the client sees 413."""
    reads = []

    async def swallowing_app(scope, receive, send):
        try:
            while (message := await receive())["type"] == "http.request":
                reads.append(len(message.get("body", b"")))
                if not message.get("more_body"):
                    break
        except Exception as error:
            await JSONResponse({"error": str(error)}, status_code=500)(
                scope, receive, send
            )
            # Reading on after the failure must not send a second 413.
            with contextlib.suppress(Exception):
                await receive()
            return
        await JSONResponse({}, status_code=200)(scope, receive, send)

    sent = await _call(
        BodySizeLimitMiddleware(swallowing_app),
        _scope("/mcp"),
        _chunks(*[b"x" * MiB] * 11),
    )

    assert _statuses(sent) == [413]
    # What the app saw stops short of the cap.
    assert sum(reads) <= MAX_BODY_BYTES


async def test_a_body_within_the_cap_reaches_the_app_whole() -> None:
    received = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            received.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        await JSONResponse({}, status_code=200)(scope, receive, send)

    parts = [b"a" * MiB, b"b" * MiB, b"c"]
    sent = await _call(BodySizeLimitMiddleware(app), _scope("/graphs"), _chunks(*parts))

    assert _statuses(sent) == [200]
    assert received == parts


def test_uploads_get_room_for_their_file(mocker: pytest_mock.MockFixture) -> None:
    mocker.patch.object(body_limit.settings.config, "max_file_size_mb", 100)

    assert max_body_bytes("/files/upload", multipart=True) == 101 * MiB
    assert max_body_bytes("/marketplace/submissions/media", multipart=True) == 11 * MiB
    assert max_body_bytes("/files/upload", multipart=False) == 0
    assert max_body_bytes("/marketplace/submissions/media", multipart=False) == 0
    assert max_body_bytes("/graphs", multipart=False) == MAX_BODY_BYTES
    assert max_body_bytes("/mcp", multipart=False) == MAX_BODY_BYTES


def test_every_route_that_parses_a_form_is_an_upload_the_cap_knows() -> None:
    """A moved upload route would refuse its uploads at the JSON cap, and a new
    one the cap does not know would take a url-encoded form."""
    routes = [r for r in v2_app.routes if isinstance(r, fastapi.routing.APIRoute)]
    form_routes = {
        r.path
        for r in routes
        if r.body_field and isinstance(r.body_field.field_info, params.Form)
    }

    assert form_routes == {
        r.path for r in routes if max_body_bytes(r.path, multipart=False) == 0
    }
    assert form_routes == {"/files/upload", "/marketplace/submissions/media"}


async def test_an_upload_that_is_not_multipart_is_refused_before_the_app_runs() -> None:
    called = []

    async def app(scope, receive, send):
        called.append(scope)

    sent = await _call(
        BodySizeLimitMiddleware(app),
        _scope(
            "/files/upload",
            headers=[(b"content-length", b"2")],
            content_type=URLENCODED,
        ),
        _chunks(b"a&"),
    )

    assert not called
    assert _statuses(sent) == [415]
    error = _body(sent)["error"]
    assert error["code"] == "unsupported_media_type"
    assert error["message"] == MULTIPART_REQUIRED


async def test_an_upload_sent_in_chunks_that_is_not_multipart_is_refused_unparsed(
    mocker: pytest_mock.MockFixture,
) -> None:
    parsed_field = mocker.spy(FormParser, "on_field_end")
    reached = []
    api = fastapi.FastAPI()

    @api.post("/files/upload")
    async def upload(file: fastapi.UploadFile) -> dict:
        reached.append(file)
        return {}

    sent = await _call(
        BodySizeLimitMiddleware(api),
        _scope("/files/upload", content_type=URLENCODED),
        _chunks(b"a&" * 1024, b"a&" * 1024),
    )

    assert not reached
    assert _statuses(sent) == [415]
    assert parsed_field.call_count == 0


@pytest.mark.parametrize("path", ["/files/upload", "/marketplace/submissions/media"])
def test_the_v2_app_refuses_an_upload_that_is_not_multipart_before_the_rate_limiter(
    path: str, mocker: pytest_mock.MockFixture
) -> None:
    """Ahead of the rate limiter, as a body over the cap is."""
    mocker.patch.object(
        global_rate_limit._anonymous_limiter,
        "check",
        side_effect=AssertionError("rate-limited before the body was refused"),
    )
    parsed_field = mocker.spy(FormParser, "on_field_end")

    response = TestClient(v2_app).post(
        path, content=b"a&" * 1024, headers={"Content-Type": URLENCODED}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert parsed_field.call_count == 0


async def test_the_cap_follows_the_path_inside_the_mounted_app(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch.object(body_limit.settings.config, "max_file_size_mb", 100)
    called = []

    async def app(scope, receive, send):
        called.append(scope)
        await JSONResponse({}, status_code=201)(scope, receive, send)

    upload = _scope(
        "/api/v2/files/upload",
        headers=[(b"content-length", str(50 * MiB).encode())],
        content_type=MULTIPART,
    )
    upload["root_path"] = "/api/v2"
    sent = await _call(BodySizeLimitMiddleware(app), upload, _chunks(b""))

    assert called
    assert _statuses(sent) == [201]


def test_the_v2_app_refuses_an_oversized_body_with_its_security_headers(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Registered inside the security headers, and ahead of the rate limiter."""
    mocker.patch.object(
        global_rate_limit._anonymous_limiter,
        "check",
        side_effect=AssertionError("rate-limited before the size was checked"),
    )

    response = TestClient(v2_app).post(
        "/graphs",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert response.headers["x-content-type-options"] == "nosniff"


def _scope(
    path: str,
    headers: list[tuple[bytes, bytes]] | None = None,
    content_type: str = "application/json",
) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode()), *(headers or [])],
        "client": ("10.0.0.9", 4242),
        "server": ("testserver", 80),
    }


def _chunks(*parts: bytes):
    """A receive that delivers *parts* as one body, then reports a disconnect."""
    pending = list(parts)

    async def receive() -> dict:
        if not pending:
            return {"type": "http.disconnect"}
        body = pending.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    return receive


async def _call(app, scope: dict, receive) -> list[dict]:
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _statuses(sent: list[dict]) -> list[int]:
    return [m["status"] for m in sent if m["type"] == "http.response.start"]


def _body(sent: list[dict]) -> dict:
    return json.loads(
        b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    )
