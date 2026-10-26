import re

import pytest
from starlette.routing import BaseRoute, Match, Mount, Route, WebSocketRoute
from starlette.types import Message, Receive, Scope, Send

from backend.api import rest_api
from backend.api.external.v2.app import v2_app
from backend.api.middleware.legacy_paths import LegacyAPIPathMiddleware


@pytest.mark.parametrize("enabled", [True, False])
def test_metrics_get_a_listener_of_their_own_only_when_enabled(
    enabled: bool, mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENABLE_METRICS", "true" if enabled else "false")
    # run() turns asyncio debug on under a local app_env; restored afterwards.
    monkeypatch.setenv("PYTHONASYNCIODEBUG", "")
    serve_api = mocker.patch("backend.api.rest_api.uvicorn.run")
    serve_metrics = mocker.patch("backend.monitoring.instrumentation.start_http_server")

    rest_api.AgentServer().run()

    config = rest_api.settings.config
    assert serve_api.call_args.kwargs["port"] == config.agent_api_port
    if not enabled:
        serve_metrics.assert_not_called()
        return
    serve_metrics.assert_called_once_with(
        config.agent_api_metrics_port, addr=config.pyro_host
    )
    assert config.agent_api_metrics_port != config.agent_api_port


def test_no_internal_route_takes_a_versioned_api_path() -> None:
    """`/api/v<n>` belongs to the public API: v2 is mounted at `/api/v2`."""
    taken = [
        route.path
        for route in rest_api.app.routes
        if isinstance(route, (Route, Mount, WebSocketRoute))
        and route is not _v2_mount()
        and re.match(r"^/api/v\d", route.path)
    ]

    assert taken == []


async def test_every_v2_path_reaches_v2_through_the_host_app() -> None:
    """An internal route with a path parameter could otherwise answer a v2
    request, such as `/api/{id}/agents` for `/api/v2/agents`, and so could the
    rewrite of the internal API's old paths."""
    v2_mount = _v2_mount()
    paths = list(_concrete_paths(v2_app.routes))
    assert len(paths) > 50

    for method, path in paths:
        scope = await _past_the_legacy_path_rewrite(
            {"type": "http", "method": method, "path": f"/api/v2{path}"}
        )
        first = next(
            route
            for route in rest_api.app.routes
            if route.matches(scope)[0] is Match.FULL
        )
        assert first is v2_mount, f"{method} /api/v2{path} reaches {first.path}"


async def _past_the_legacy_path_rewrite(scope: Scope) -> Scope:
    seen: list[Scope] = []

    async def record(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope)

    installed = next(
        m for m in rest_api.app.user_middleware if m.cls is LegacyAPIPathMiddleware
    )
    await installed.cls(record, **installed.kwargs)(scope, _no_body, _no_send)
    return seen[0]


async def _no_body() -> Message:
    raise AssertionError("routing a path reads no body")


async def _no_send(message: Message) -> None:
    raise AssertionError("routing a path sends nothing")


def _v2_mount() -> Mount:
    return next(
        route
        for route in rest_api.app.routes
        if isinstance(route, Mount) and route.app is v2_app
    )


def _concrete_paths(routes: list[BaseRoute]):
    for route in routes:
        if isinstance(route, Mount):
            yield "GET", route.path + "/"
        elif isinstance(route, Route):
            for method in route.methods or []:
                yield method, re.sub(r"{[^}]+}", "x", route.path)
