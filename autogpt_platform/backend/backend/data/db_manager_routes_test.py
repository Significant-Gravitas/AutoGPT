"""Pins the database manager's RPC surface: every route a caller in another
process can reach, with its parameters and return type, and which routes each
client offers. A deploy runs old executors against a new manager and the other
way round, so a route that is dropped, renamed or re-typed breaks calls in flight."""

import inspect
import json
import types
import typing
from collections.abc import Callable
from typing import Any

from pytest_snapshot.plugin import Snapshot

from backend.util.service import EXPOSED_FLAG, AppServiceClient, resolve_type_hints

from .db_manager import (
    DatabaseManager,
    DatabaseManagerAsyncClient,
    DatabaseManagerClient,
)


def test_the_rpc_surface_matches_the_snapshot(snapshot: Snapshot) -> None:
    """Update with ``--snapshot-update`` only for a route that is new: changing
    or removing one needs both sides of the RPC to cope during a rollout."""
    surface = {
        "routes": {
            name: _signature(endpoint) for name, endpoint in _server_routes().items()
        },
        "async_client": _client_routes(DatabaseManagerAsyncClient),
        "sync_client": _client_routes(DatabaseManagerClient),
    }

    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(surface, indent=1, sort_keys=True) + "\n", "db_manager_rpc_surface"
    )


def test_every_client_method_calls_a_route_with_the_signature_it_shows() -> None:
    """The client builds each request from its own copy of the signature."""
    routes = _server_routes()
    mismatched = [
        f"{client.__name__}.{name}"
        for client in (DatabaseManagerAsyncClient, DatabaseManagerClient)
        for name, method in _client_methods(client).items()
        if method.__name__ not in routes
        or _signature(method) != _signature(routes[method.__name__])
    ]

    assert not mismatched


def test_the_async_client_awaits_and_the_sync_client_blocks() -> None:
    """The client picks its transport by whether the method is a coroutine."""
    assert all(
        inspect.iscoroutinefunction(method)
        for method in _client_methods(DatabaseManagerAsyncClient).values()
    )
    assert not any(
        inspect.iscoroutinefunction(method)
        for method in _client_methods(DatabaseManagerClient).values()
    )


def _server_routes() -> dict[str, Callable[..., Any]]:
    """Read the way ``AppService`` registers routes: the class's own exposed attributes."""
    return {
        name: attr
        for name, attr in vars(DatabaseManager).items()
        if getattr(attr, EXPOSED_FLAG, False)
    }


def _client_routes(client: type[AppServiceClient]) -> dict[str, str]:
    """Method name to the route it posts to, which is the method's ``__name__``."""
    return {name: method.__name__ for name, method in _client_methods(client).items()}


def _client_methods(client: type[AppServiceClient]) -> dict[str, Callable[..., Any]]:
    return {
        name: attr
        for name, attr in vars(client).items()
        if inspect.isfunction(attr) and not name.startswith("_")
    }


def _signature(func: Callable[..., Any]) -> dict[str, Any]:
    hints = resolve_type_hints(func)
    signature = inspect.signature(func)
    return {
        "params": [
            _render_param(param, hints.get(param.name, param.annotation))
            for param in signature.parameters.values()
            if param.name not in ("self", "cls")
        ],
        "returns": _render_type(hints.get("return", signature.return_annotation)),
    }


def _render_param(param: inspect.Parameter, annotation: Any) -> str:
    rendered = f"{param.name}: {_render_type(annotation)}"
    if param.kind is inspect.Parameter.KEYWORD_ONLY:
        rendered = f"*, {rendered}"
    if param.default is not inspect.Parameter.empty:
        rendered += f" = {_render_default(param.default)}"
    return rendered


def _render_default(default: Any) -> str:
    if default is None or isinstance(default, (bool, int, float, str)):
        return repr(default)
    return f"<{type(default).__name__}>"


def _render_type(annotation: Any) -> str:
    """One spelling per type on every Python version: ``Optional[X]``,
    ``Union[X, None]`` and ``X | None`` all render as ``X | None``."""
    if annotation is inspect.Parameter.empty:
        return "<unannotated>"
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, typing.ForwardRef):
        return annotation.__forward_arg__

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return " | ".join(_render_type(arg) for arg in args)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(arg) for arg in args)}]"
    if origin is typing.Annotated:
        return f"Annotated[{_render_type(args[0])}, ...]"
    if origin is not None:
        rendered_args = ", ".join(
            "..." if arg is Ellipsis else _render_type(arg) for arg in args
        )
        return f"{_render_type(origin)}[{rendered_args}]"
    if isinstance(annotation, type):
        if annotation.__module__ == "builtins":
            return annotation.__qualname__
        return f"{annotation.__module__}.{annotation.__qualname__}"
    return repr(annotation)
