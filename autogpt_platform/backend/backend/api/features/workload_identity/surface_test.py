"""Pins the published workload-identity surface: paths, operation IDs and tags.

Operation IDs are built from each route's summary and first tag, so renaming
either silently renames the generated frontend client's method.
"""

import pytest

from backend.api.rest_api import app

EXPECTED_OPERATIONS = {
    ("get", "/api/workload-identities"): "getV1List workload identities",
    ("post", "/api/workload-identities"): "postV1Trust a workload identity",
    (
        "get",
        "/api/workload-identities/{identity_id}",
    ): "getV1Get a workload identity",
    (
        "patch",
        "/api/workload-identities/{identity_id}",
    ): "patchV1Enable or disable a workload identity",
    (
        "delete",
        "/api/workload-identities/{identity_id}",
    ): "deleteV1Revoke a workload identity",
}


@pytest.mark.parametrize(
    "method,path,operation_id",
    [(m, p, oid) for (m, p), oid in EXPECTED_OPERATIONS.items()],
)
def test_workload_identity_operation_is_published(method, path, operation_id):
    operation = app.openapi()["paths"][path][method]
    assert operation["operationId"] == operation_id
    assert operation["tags"] == ["v1", "workload-identity"]


def test_workload_identity_surface_has_no_other_operations():
    published = {
        (method, path)
        for path, operations in app.openapi()["paths"].items()
        if path.startswith("/api/workload-identities")
        for method in operations
    }
    assert published == set(EXPECTED_OPERATIONS)


def test_the_token_endpoint_documents_the_exchange_grant():
    schema = app.openapi()
    body = schema["paths"]["/api/oauth/token"]["post"]["requestBody"]
    refs = str(body["content"]["application/json"]["schema"])
    assert "TokenRequestByTokenExchange" in refs
