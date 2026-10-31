import ast
import inspect
import logging
import re
import textwrap
from unittest.mock import AsyncMock

import pytest
from starlette.routing import BaseRoute, Match, Mount, Route, WebSocketRoute

from backend.api import rest_api
from backend.api.external.v2.app import v2_app


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


def test_every_v2_path_reaches_v2_through_the_host_app() -> None:
    """An internal route with a path parameter could otherwise answer a v2
    request, such as `/api/{id}/agents` for `/api/v2/agents`."""
    v2_mount = _v2_mount()
    paths = list(_concrete_paths(v2_app.routes))
    assert len(paths) > 50

    for method, path in paths:
        scope = {"type": "http", "method": method, "path": f"/api/v2{path}"}
        first = next(
            route
            for route in rest_api.app.routes
            if route.matches(scope)[0] is Match.FULL
        )
        assert first is v2_mount, f"{method} /api/v2{path} reaches {first.path}"


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


# Every step the lifespan awaits before it serves. A failure in any of these
# stops the boot, so a data migration goes in `_run_data_migrations` instead.
_LIFESPAN_STARTUP_AWAITS = {
    "backend.data.db.connect",
    "backend.data.redis_client.get_redis_async",
    "backend.data.block.initialize_blocks",
    "_run_data_migrations",
}

_DATA_MIGRATIONS = {
    "user integrations": "backend.data.user.migrate_and_encrypt_user_integrations",
    "LLM models": "backend.data.graph.migrate_llm_models",
    "legacy triggered graphs": (
        "backend.integrations.webhooks.utils.migrate_legacy_triggered_graphs"
    ),
    "org bootstrap": "backend.data.org_migration.run_migration",
    "Otto transport": "backend.data.autopilot_migrate.migrate_autopilot_transport",
}


def test_lifespan_awaits_only_the_start_up_steps_allowed_to_stop_the_boot() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(rest_api.lifespan_context.__wrapped__))
    )
    serving_from = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Yield))
    awaited = {
        ast.unparse(n.value.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Await)
        and isinstance(n.value, ast.Call)
        and n.lineno < serving_from
    }

    assert awaited == _LIFESPAN_STARTUP_AWAITS, (
        "A data migration awaited in the lifespan stops the boot when it fails; "
        "add it to `_run_data_migrations`, or to this list if it must."
    )


async def test_a_failed_data_migration_is_logged_and_the_rest_still_run(
    mocker, caplog
) -> None:
    migrations = {
        name: mocker.patch(target, new_callable=AsyncMock)
        for name, target in _DATA_MIGRATIONS.items()
    }
    migrations["org bootstrap"].side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger=rest_api.logger.name):
        await rest_api._run_data_migrations()

    for migrate in migrations.values():
        migrate.assert_awaited_once()
    assert [
        r.getMessage() for r in caplog.records if r.name == rest_api.logger.name
    ] == ["Start-up data migration 'org bootstrap' failed"]
