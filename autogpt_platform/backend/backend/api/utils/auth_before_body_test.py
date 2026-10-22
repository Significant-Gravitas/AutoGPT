import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest
from autogpt_libs.auth.dependencies import (
    get_optional_user_id,
    get_user_id,
    requires_user,
)
from autogpt_libs.auth.jwt_utils import bearer_jwt_auth, get_jwt_payload
from fastapi import FastAPI, Request, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.types import ASGIApp

from backend.api import rest_api
from backend.api.utils.auth_before_body import check_token_before_body

MULTIPART = "multipart/form-data; boundary=x"
JSON = {"content-type": "application/json"}


class Item(BaseModel):
    name: str


def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/signed-in")
    async def signed_in(item: Item, payload: dict = Security(get_jwt_payload)):
        return {"sub": payload["sub"], "name": item.name}

    @app.post("/upload")
    async def upload(file: UploadFile, payload: dict = Security(get_jwt_payload)):
        return {"size": len(await file.read())}

    @app.post("/route-dependency", dependencies=[Security(requires_user)])
    async def route_dependency(item: Item):
        return {"name": item.name}

    @app.post("/user-id")
    async def with_user_id(item: Item, user_id: str = Security(get_user_id)):
        return {"user_id": user_id}

    @app.post("/optional")
    async def optional(
        item: Item, user_id: str | None = Security(get_optional_user_id)
    ):
        return {"user_id": user_id, "name": item.name}

    @app.post("/public")
    async def public(item: Item):
        return {"name": item.name}

    check_token_before_body(app)
    return app


async def _post(
    app: ASGIApp,
    path: str,
    body: bytes,
    headers: dict[str, str],
    disconnect: bool = False,
):
    """Call the app as uvicorn would. Besides the response, returns how much of
    the body the app received and whether it parsed it (body, JSON or form).
    With `disconnect`, the client goes away instead of sending its body."""
    received = 0

    async def receive():
        nonlocal received
        if disconnect:
            return {"type": "http.disconnect"}
        received += len(body)
        return {"type": "http.request", "body": body, "more_body": False}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    parsed: list[str] = []

    def spy(name: str):
        real = getattr(Request, name)

        def parse(self, *args, **kwargs):
            parsed.append(name)
            return real(self, *args, **kwargs)

        return parse

    framing = (
        {} if "transfer-encoding" in headers else {"content-length": str(len(body))}
    )
    all_headers = {**framing, **headers}
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
        "headers": [(k.encode(), v.encode()) for k, v in all_headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    with ExitStack() as stack:
        for name in ("body", "json", "form"):
            stack.enter_context(patch.object(Request, name, spy(name)))
        try:
            await app(scope, receive, send)
        except Exception:
            # Starlette answers an unhandled error with a 500, then re-raises.
            if not sent:
                raise
    start = next(m for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent[1:])
    is_json = content.startswith(b"{")
    return (
        start["status"],
        {k.decode(): v.decode() for k, v in start["headers"]},
        json.loads(content) if is_json else content.decode(),
        received == len(body),
        bool(parsed),
    )


@pytest.mark.parametrize(
    "path, body, content_type",
    [
        ("/signed-in", b'{"name": "x"}', "application/json"),
        # Not even valid JSON: refused for the token before the body is parsed.
        ("/signed-in", b"{not json", "application/json"),
        ("/upload", b"--x\r\n" + b"A" * 4096, MULTIPART),
        ("/route-dependency", b'{"name": "x"}', "application/json"),
        ("/user-id", b'{"name": "x"}', "application/json"),
    ],
    ids=["json", "malformed-json", "upload", "route-dependency", "user-id"],
)
async def test_a_request_without_a_token_is_refused_before_its_body_is_parsed(
    path: str, body: bytes, content_type: str
):
    status, headers, payload, drained, parsed = await _post(
        _app(), path, body, {"content-type": content_type}
    )

    assert status == 401
    assert payload == {"detail": "Authorization header is missing"}
    assert parsed is False
    # Received and thrown away, with the connection left as the client chose,
    # so the client reads the 401 instead of a reset.
    assert drained is True
    assert "connection" not in headers


async def test_a_chunked_body_without_a_token_is_refused_before_it_is_parsed():
    status, _, _, drained, parsed = await _post(
        _app(),
        "/upload",
        b"--x\r\n" + b"A" * 4096,
        {"content-type": MULTIPART, "transfer-encoding": "chunked"},
    )

    assert (status, drained, parsed) == (401, True, False)


async def test_a_client_that_leaves_mid_upload_is_still_answered_as_refused():
    """Not a 500: the disconnect while the body is thrown away is expected."""
    status, _, payload, _, parsed = await _post(
        _app(), "/upload", b"--x\r\n" + b"A" * 4096, {"content-type": MULTIPART}, True
    )

    assert (status, payload, parsed) == (
        401,
        {"detail": "Authorization header is missing"},
        False,
    )


async def test_an_unexpected_error_in_the_check_still_drains_the_body():
    """A 500 sent with the body unread could reset the connection as well."""
    with patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(side_effect=RuntimeError("key type mismatch")),
    ):
        status, _, _, drained, parsed = await _post(
            _app(),
            "/signed-in",
            b'{"name": "x"}',
            {**JSON, "authorization": "Bearer odd"},
        )

    assert (status, drained, parsed) == (500, True, False)


async def test_a_bad_token_gets_the_dependencys_own_401_before_the_body_is_parsed():
    with patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(side_effect=ValueError("Invalid token: bad signature")),
    ):
        status, _, payload, _, parsed = await _post(
            _app(),
            "/signed-in",
            b'{"name": "x"}',
            {**JSON, "authorization": "Bearer nope"},
        )

    assert status == 401
    assert payload == {"detail": "Invalid token: bad signature"}
    assert parsed is False


async def test_a_valid_token_reaches_the_route_with_its_body():
    with patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(return_value={"sub": "user-1"}),
    ):
        status, _, payload, _, parsed = await _post(
            _app(),
            "/signed-in",
            b'{"name": "x"}',
            {**JSON, "authorization": "Bearer good"},
        )

    assert (status, payload, parsed) == (200, {"sub": "user-1", "name": "x"}, True)


@pytest.mark.parametrize("path", ["/public", "/optional"])
async def test_routes_open_to_anonymous_callers_still_parse_the_body(path: str):
    status, _, payload, _, parsed = await _post(_app(), path, b'{"name": "x"}', JSON)

    assert (status, payload["name"], parsed) == (200, "x", True)


async def test_an_overridden_sign_in_dependency_is_used_as_fastapi_would():
    """Tests override the sign-in dependency instead of minting tokens; the
    early check must then defer to the override, as the route does."""
    app = _app()
    app.dependency_overrides[get_user_id] = lambda: "test-user"

    status, _, payload, _, _ = await _post(app, "/user-id", b'{"name": "x"}', JSON)
    assert (status, payload) == (200, {"user_id": "test-user"})

    # Overriding one route's dependency doesn't open another route.
    status, _, _, _, parsed = await _post(app, "/signed-in", b'{"name": "x"}', JSON)
    assert (status, parsed) == (401, False)


async def test_an_overridden_bearer_scheme_is_used_as_fastapi_would():
    """The early check runs the real bearer scheme, so with it overridden the
    route is left to FastAPI, which reads the credentials from the override."""
    app = _app()
    app.dependency_overrides[bearer_jwt_auth] = lambda: HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="from-the-override"
    )

    with patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(return_value={"sub": "user-1"}),
    ):
        status, _, payload, _, parsed = await _post(
            app, "/signed-in", b'{"name": "x"}', JSON
        )

    assert (status, payload, parsed) == (200, {"sub": "user-1", "name": "x"}, True)


async def test_an_override_that_itself_needs_a_token_is_checked_early():
    """FastAPI resolves an override's own dependencies, so a replacement that
    still needs a token gets the early check too."""

    async def replacement(payload: dict = Security(get_jwt_payload)) -> str:
        return payload["sub"]

    app = _app()
    app.dependency_overrides[get_user_id] = replacement

    status, _, _, _, parsed = await _post(app, "/user-id", b'{"name": "x"}', JSON)
    assert (status, parsed) == (401, False)


@pytest.mark.parametrize(
    "path, body, content_type",
    [
        ("/api/graphs", b"{not json", "application/json"),
        ("/api/files/upload", b"--x\r\n" + b"A" * 4096, MULTIPART),
    ],
    ids=["graph-json", "file-upload"],
)
async def test_the_platform_api_checks_the_token_before_the_body(
    path: str, body: bytes, content_type: str
):
    status, _, payload, drained, parsed = await _post(
        rest_api.app, path, body, {"content-type": content_type}
    )

    assert status == 401
    assert payload == {"detail": "Authorization header is missing"}
    assert (drained, parsed) == (True, False)


async def test_the_early_401_carries_the_cors_headers_a_browser_needs(
    mocker, monkeypatch: pytest.MonkeyPatch
):
    """A browser can read the 401, and refresh its token, only if the response
    carries the CORS headers AgentServer wraps around the app."""
    # run() turns asyncio debug on under a local app_env; restored afterwards.
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "")
    serve = mocker.patch("backend.api.rest_api.uvicorn.run")
    rest_api.AgentServer().run()
    served = serve.call_args.kwargs["app"]
    origin = rest_api.settings.config.backend_cors_allow_origins[0]

    status, headers, _, _, parsed = await _post(
        served, "/api/graphs", b'{"name": "x"}', {**JSON, "origin": origin}
    )

    assert (status, parsed) == (401, False)
    assert headers["access-control-allow-origin"] == origin


async def test_the_platform_api_still_parses_a_signed_in_users_body():
    """Past the early check, a valid token reaches the route's own handling:
    here, its body validation."""
    with patch(
        "autogpt_libs.auth.jwt_utils.parse_jwt_token_async",
        AsyncMock(return_value={"sub": "user-1", "role": "authenticated"}),
    ):
        status, _, _, _, parsed = await _post(
            rest_api.app,
            "/api/graphs",
            b"{not json",
            {**JSON, "authorization": "Bearer good"},
        )

    assert (status, parsed) == (422, True)
