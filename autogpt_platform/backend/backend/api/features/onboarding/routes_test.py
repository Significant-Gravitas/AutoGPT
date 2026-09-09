"""Pins the mounted onboarding surface."""

import pytest
from fastapi.routing import APIRoute

from backend.api.rest_api import app as real_app

# Tags stay per-route: /onboarding/completed publishes ["onboarding", "public"]
# while the other six publish ["onboarding"], so neither can hoist.
EXPECTED_OPERATIONS = {
    ("get", "/api/onboarding"): ["v1", "onboarding"],
    ("patch", "/api/onboarding"): ["v1", "onboarding"],
    ("post", "/api/onboarding/step"): ["v1", "onboarding"],
    ("get", "/api/onboarding/agents"): ["v1", "onboarding"],
    ("get", "/api/onboarding/completed"): ["v1", "onboarding", "public"],
    ("post", "/api/onboarding/reset"): ["v1", "onboarding"],
    ("post", "/api/onboarding/profile"): ["v1", "onboarding"],
}


@pytest.mark.parametrize(
    "method,path,tags", [(m, p, t) for (m, p), t in EXPECTED_OPERATIONS.items()]
)
def test_onboarding_operation_is_published(method: str, path: str, tags: list[str]):
    assert real_app.openapi()["paths"][path][method]["tags"] == tags


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_onboarding_route_requires_an_authenticated_user(path: str):
    """`security` in the schema does not prove this — each handler's own
    Security(get_user_id) puts it there. Assert the dependency. The "public"
    tag on /onboarding/completed names its audience, not its auth."""
    for route in real_app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            assert "requires_user" in {
                d.call.__name__ for d in route.dependant.dependencies if d.call
            }


def test_onboarding_surface_has_no_other_operations():
    """Keyed on the owning module: /api/onboarding/brain-dump/* shares the
    prefix but belongs to features/onboarding_dump."""
    served = {
        (method.lower(), route.path)
        for route in real_app.routes
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == "backend.api.features.onboarding.routes"
        for method in route.methods
        if method != "HEAD"
    }
    assert served == set(EXPECTED_OPERATIONS)
