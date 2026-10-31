"""The platform API bounds every request body by the route it reaches, the
mounted apps' routes included."""

import json
import re
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from starlette.routing import Mount

from backend.api import rest_api
from backend.api.external.v2 import global_rate_limit
from backend.api.features.integrations.router import WEBHOOK_PAYLOAD_LIMIT
from backend.api.middleware.body_limit import (
    MiB,
    route_body_limit,
    settings,
    upload_body_limit,
)

MULTIPART = "multipart/form-data; boundary=x"
JSON = "application/json"
CHUNKED = {"transfer-encoding": "chunked"}
VALID_TOKEN = {"authorization": "Bearer good"}

# Every server the backend starts, and why its request bodies are bounded.
SERVERS = {
    "backend/api/rest_api.py": "RequestBodyLimitMiddleware on the app",
    "backend/api/ws_api.py": "websockets only; no route takes a body",
    "backend/util/service.py": "internal; the service key is checked first",
}


async def test_a_signed_in_body_over_the_limit_is_refused_unparsed():
    limit = _default_limit()
    with _signed_in():
        sent = await _post(
            "/api/graphs",
            [b"{", b" " * limit, b"}"],
            {"content-type": JSON, **CHUNKED, **VALID_TOKEN},
        )

    assert sent.status == 413
    assert not sent.parsed
    assert sent.received <= limit + 1


async def test_a_declared_length_over_the_limit_is_refused_before_any_byte_is_read():
    """On the mounted external API, as on the app's own routes."""
    sent = await _post(
        "/external-api/v1/blocks/0/execute",
        [b"{}"],
        {"content-type": JSON, "content-length": str(_default_limit() + 1)},
    )

    assert sent.status == 413
    assert sent.received == 0


async def test_an_upload_route_takes_a_multipart_body_above_the_default_limit():
    """Refused for its missing token, not its size: the whole body was let through."""
    body = [b"--x\r\n", b"A" * (_default_limit() + MiB)]

    sent = await _post("/api/files/upload", body, {"content-type": MULTIPART})

    assert sent.status == 401
    assert sent.received == sum(map(len, body))


async def test_an_upload_route_holds_any_other_body_to_the_default_limit():
    sent = await _post(
        "/api/files/upload",
        [b"a=" + b"A" * (_default_limit() + MiB)],
        {"content-type": "application/x-www-form-urlencoded"},
    )

    assert sent.status == 413


async def test_a_mounted_app_that_bounds_its_own_bodies_answers_in_its_own_shape():
    """v2 counts the body itself; the platform limit stands aside for it."""

    async def no_rate_limit(self, scope, receive, send):
        await self.app(scope, receive, send)

    with patch.object(
        global_rate_limit.GlobalRateLimitMiddleware, "__call__", no_rate_limit
    ):
        sent = await _post(
            "/api/v2/graphs", [b" " * MiB] * 11, {"content-type": JSON, **CHUNKED}
        )

    assert sent.status == 413
    assert json.loads(sent.content)["error"]["code"] == "payload_too_large"


@pytest.mark.parametrize(
    "path",
    [
        "/api/files/upload",
        "/api/workspace/files/upload",
        "/api/store/submissions/media",
        "/api/skills/package",
        "/api/oauth/apps/{app_id}/logo/upload",
        "/api/onboarding/brain-dump/parts",
    ],
)
def test_every_upload_route_gets_room_for_its_file(path: str):
    """A route whose file parameter isn't recognised would refuse its uploads."""
    route = _routes()[path]

    assert route_body_limit(_matched(route, MULTIPART)) == upload_body_limit()
    assert route_body_limit(_matched(route, JSON)) == _default_limit()


def test_webhook_ingress_takes_the_largest_provider_payload():
    route = _routes()["/api/integrations/{provider}/webhooks/{webhook_id}/ingress"]

    assert route_body_limit(_matched(route, JSON)) == WEBHOOK_PAYLOAD_LIMIT


def test_every_server_the_backend_starts_is_accounted_for():
    """A new server would take bodies without the platform API's limit."""
    backend = Path(__file__).parents[2]
    starts = {
        path.relative_to(backend.parent).as_posix()
        for path in backend.rglob("*.py")
        if not path.name.endswith("_test.py")
        and re.search(r"\buvicorn\.(run|Server)\(", path.read_text())
    }

    assert starts == set(SERVERS)


class _Sent:
    def __init__(self, status: int, content: bytes, received: int, parsed: bool):
        self.status = status
        self.content = content
        self.received = received
        self.parsed = parsed


async def _post(path: str, chunks: list[bytes], headers: dict[str, str]) -> _Sent:
    """Call the platform API as uvicorn would, the body in `chunks`. Returns
    the answer, how much body the app read and whether FastAPI parsed it."""
    if "transfer-encoding" not in headers:
        headers = {"content-length": str(sum(map(len, chunks))), **headers}
    pending = list(chunks)
    received = 0

    async def receive():
        nonlocal received
        if not pending:
            return {"type": "http.disconnect"}
        body = pending.pop(0)
        received += len(body)
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    parsed: list[str] = []
    with ExitStack() as stack:
        for name in ("json", "form"):
            stack.enter_context(patch.object(Request, name, _spy(name, parsed)))
        await rest_api.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent if "body" in m)
    return _Sent(start["status"], content, received, bool(parsed))


def _spy(name: str, parsed: list[str]):
    real = getattr(Request, name)

    def parse(self, *args, **kwargs):
        parsed.append(name)
        return real(self, *args, **kwargs)

    return parse


def _signed_in():
    return patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(return_value={"sub": "user-1", "role": "authenticated"}),
    )


def _default_limit() -> int:
    return settings.config.max_request_body_mb * MiB


def _routes() -> dict[str, APIRoute]:
    return dict(_walk(rest_api.app.routes, ""))


def _walk(routes, prefix: str):
    for route in routes:
        if isinstance(route, Mount):
            yield from _walk(route.routes, prefix + route.path)
        elif isinstance(route, APIRoute):
            yield prefix + route.path, route


def _matched(route: APIRoute, content_type: str) -> dict:
    return {"route": route, "headers": [(b"content-type", content_type.encode())]}
