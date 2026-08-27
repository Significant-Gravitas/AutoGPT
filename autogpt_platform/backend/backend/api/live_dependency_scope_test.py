from collections import Counter
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from backend.api.external.fastapi_app import external_api
from backend.api.rest_api import app


def _walk_dependencies(dependant: Dependant) -> Iterator[Dependant]:
    for child in dependant.dependencies:
        yield child
        yield from _walk_dependencies(child)


def _live_dependency_kind(dependant: Dependant) -> str | None:
    call = dependant.call
    module = getattr(call, "__module__", "")
    name = getattr(call, "__qualname__", "")
    if module == "backend.api.live_auth" and name.startswith("requires_live_"):
        return "live auth"
    if (
        module == "backend.api.external.middleware"
        and name == "permission_dependency.<locals>.check_permissions"
    ):
        return "external auth"
    if module == "backend.api.features.library.routes.live" and name.startswith(
        "require_live_library_"
    ):
        return "library mutation"
    if module == "backend.api.features.v1" and name.startswith("_live_graph_"):
        return "graph mutation"
    if (
        module == "backend.api.features.executions.review.routes"
        and name == "_live_review_action_dependency"
    ):
        return "review mutation"
    return None


def _live_dependencies(
    *apps: FastAPI,
) -> Iterator[tuple[str, str, Dependant]]:
    for target_app in apps:
        for route in target_app.routes:
            if not isinstance(route, APIRoute):
                continue
            for dependant in _walk_dependencies(route.dependant):
                if kind := _live_dependency_kind(dependant):
                    yield route.path, kind, dependant


def test_all_transaction_holding_dependencies_are_function_scoped() -> None:
    live_dependencies = list(_live_dependencies(app, external_api))
    kind_counts = Counter(kind for _, kind, _ in live_dependencies)
    wrong_scope = [
        (path, kind, getattr(dependant.call, "__qualname__", ""), dependant.scope)
        for path, kind, dependant in live_dependencies
        if dependant.scope != "function"
    ]

    assert kind_counts >= Counter(
        {
            "live auth": 79,
            "external auth": 15,
            "library mutation": 13,
            "graph mutation": 4,
            "review mutation": 1,
        }
    )
    assert wrong_scope == []
