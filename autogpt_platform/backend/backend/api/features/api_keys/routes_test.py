"""Pins the published API-key surface: paths, operation IDs and tags are contract.

Operation IDs are built from each route's summary and first tag, so renaming
either silently renames the generated frontend client's method.
"""

import pytest
from fastapi.routing import APIRoute

from backend.api.rest_api import app

EXPECTED_OPERATIONS = {
    ("get", "/api/api-keys"): "getV1List user api keys",
    ("post", "/api/api-keys"): "postV1Create new api key",
    ("delete", "/api/api-keys/{key_id}"): "deleteV1Revoke api key",
    ("get", "/api/api-keys/{key_id}"): "getV1Get specific api key",
    ("put", "/api/api-keys/{key_id}/permissions"): "putV1Update key permissions",
    ("post", "/api/api-keys/{key_id}/suspend"): "postV1Suspend api key",
}


@pytest.mark.parametrize(
    "method,path,operation_id",
    [(m, p, oid) for (m, p), oid in EXPECTED_OPERATIONS.items()],
)
def test_api_key_operation_is_published(method: str, path: str, operation_id: str):
    operation = app.openapi()["paths"][path][method]
    assert operation["operationId"] == operation_id
    assert operation["tags"] == ["v1", "api-keys"]


def test_api_key_surface_has_no_other_operations():
    published = {
        (method, path)
        for path, operations in app.openapi()["paths"].items()
        if path.startswith("/api/api-keys")
        for method in operations
    }
    assert published == set(EXPECTED_OPERATIONS)


@pytest.mark.parametrize("path", sorted({p for _, p in EXPECTED_OPERATIONS}))
def test_api_key_path_is_served_by_this_module(path: str):
    handlers = {
        route.endpoint.__module__
        for route in app.routes
        if isinstance(route, APIRoute) and route.path == path
    }
    assert handlers == {"backend.api.features.api_keys.routes"}
