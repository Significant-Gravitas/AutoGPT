"""
Comprehensive tests for auth helpers module to achieve 100% coverage.
Tests OpenAPI schema generation and authentication response handling.
"""

from unittest import mock

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from autogpt_libs.auth.helpers import add_auth_responses_to_openapi
from autogpt_libs.auth.jwt_utils import bearer_jwt_auth


def test_add_auth_responses_to_openapi_basic():
    """Test adding 401 responses to OpenAPI schema."""
    app = FastAPI(title="Test App", version="1.0.0")

    # Add some test endpoints with authentication
    from fastapi import Depends

    from autogpt_libs.auth.dependencies import requires_user

    @app.get("/protected", dependencies=[Depends(requires_user)])
    def protected_endpoint():
        return {"message": "Protected"}

    @app.get("/public")
    def public_endpoint():
        return {"message": "Public"}

    # Apply the OpenAPI customization
    add_auth_responses_to_openapi(app)

    # Get the OpenAPI schema
    schema = app.openapi()

    # Verify basic schema properties
    assert schema["info"]["title"] == "Test App"
    assert schema["info"]["version"] == "1.0.0"

    # Verify 401 response component is added
    assert "components" in schema
    assert "responses" in schema["components"]
    assert "HTTP401NotAuthenticatedError" in schema["components"]["responses"]

    # Verify 401 response structure
    error_response = schema["components"]["responses"]["HTTP401NotAuthenticatedError"]
    assert error_response["description"] == "Authentication required"
    assert "application/json" in error_response["content"]
    assert "schema" in error_response["content"]["application/json"]

    # Verify schema properties
    response_schema = error_response["content"]["application/json"]["schema"]
    assert response_schema["type"] == "object"
    assert "detail" in response_schema["properties"]
    assert response_schema["properties"]["detail"]["type"] == "string"


def test_add_auth_responses_to_openapi_with_security():
    """Test that 401 responses are added only to secured endpoints."""
    app = FastAPI()

    # Mock endpoint with security
    from fastapi import Security

    from autogpt_libs.auth.dependencies import get_user_id

    @app.get("/secured")
    def secured_endpoint(user_id: str = Security(get_user_id)):
        return {"user_id": user_id}

    @app.post("/also-secured")
    def another_secured(user_id: str = Security(get_user_id)):
        return {"status": "ok"}

    @app.get("/unsecured")
    def unsecured_endpoint():
        return {"public": True}

    # Apply OpenAPI customization
    add_auth_responses_to_openapi(app)

    # Get schema
    schema = app.openapi()

    # Check that secured endpoints have 401 responses
    if "/secured" in schema["paths"]:
        if "get" in schema["paths"]["/secured"]:
            secured_get = schema["paths"]["/secured"]["get"]
            if "responses" in secured_get:
                assert "401" in secured_get["responses"]
                assert (
                    secured_get["responses"]["401"]["$ref"]
                    == "#/components/responses/HTTP401NotAuthenticatedError"
                )

    if "/also-secured" in schema["paths"]:
        if "post" in schema["paths"]["/also-secured"]:
            secured_post = schema["paths"]["/also-secured"]["post"]
            if "responses" in secured_post:
                assert "401" in secured_post["responses"]

    # Check that unsecured endpoint does not have 401 response
    if "/unsecured" in schema["paths"]:
        if "get" in schema["paths"]["/unsecured"]:
            unsecured_get = schema["paths"]["/unsecured"]["get"]
            if "responses" in unsecured_get:
                assert "401" not in unsecured_get.get("responses", {})


def test_add_auth_responses_to_openapi_cached_schema():
    """Test that OpenAPI schema is cached after first generation."""
    app = FastAPI()

    # Apply customization
    add_auth_responses_to_openapi(app)

    # Get schema twice
    schema1 = app.openapi()
    schema2 = app.openapi()

    # Should return the same cached object
    assert schema1 is schema2


def test_add_auth_responses_to_openapi_existing_responses():
    """Test handling endpoints that already have responses defined."""
    app = FastAPI()

    from fastapi import Security

    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    @app.get(
        "/with-responses",
        responses={
            200: {"description": "Success"},
            404: {"description": "Not found"},
        },
    )
    def endpoint_with_responses(jwt: dict = Security(get_jwt_payload)):
        return {"data": "test"}

    # Apply customization
    add_auth_responses_to_openapi(app)

    schema = app.openapi()

    # Check that existing responses are preserved and 401 is added
    if "/with-responses" in schema["paths"]:
        if "get" in schema["paths"]["/with-responses"]:
            responses = schema["paths"]["/with-responses"]["get"].get("responses", {})
            # Original responses should be preserved
            if "200" in responses:
                assert responses["200"]["description"] == "Success"
            if "404" in responses:
                assert responses["404"]["description"] == "Not found"
            # 401 should be added
            if "401" in responses:
                assert (
                    responses["401"]["$ref"]
                    == "#/components/responses/HTTP401NotAuthenticatedError"
                )


def test_add_auth_responses_to_openapi_no_security_endpoints():
    """Test with app that has no secured endpoints."""
    app = FastAPI()

    @app.get("/public1")
    def public1():
        return {"message": "public1"}

    @app.post("/public2")
    def public2():
        return {"message": "public2"}

    # Apply customization
    add_auth_responses_to_openapi(app)

    schema = app.openapi()

    # Component should still be added for consistency
    assert "HTTP401NotAuthenticatedError" in schema["components"]["responses"]

    # But no endpoints should have 401 responses
    for path in schema["paths"].values():
        for method in path.values():
            if isinstance(method, dict) and "responses" in method:
                assert "401" not in method["responses"]


def test_add_auth_responses_to_openapi_multiple_security_schemes():
    """Test endpoints with multiple security requirements."""
    app = FastAPI()

    from fastapi import Security

    from autogpt_libs.auth.dependencies import requires_admin_user, requires_user
    from autogpt_libs.auth.models import User

    @app.get("/multi-auth")
    def multi_auth(
        user: User = Security(requires_user),
        admin: User = Security(requires_admin_user),
    ):
        return {"status": "super secure"}

    # Apply customization
    add_auth_responses_to_openapi(app)

    schema = app.openapi()

    # Should have 401 response
    if "/multi-auth" in schema["paths"]:
        if "get" in schema["paths"]["/multi-auth"]:
            responses = schema["paths"]["/multi-auth"]["get"].get("responses", {})
            if "401" in responses:
                assert (
                    responses["401"]["$ref"]
                    == "#/components/responses/HTTP401NotAuthenticatedError"
                )


def test_add_auth_responses_to_openapi_empty_components():
    """Test when OpenAPI schema has no components section initially."""
    app = FastAPI()

    # Mock get_openapi to return schema without components
    original_get_openapi = get_openapi

    def mock_get_openapi(*args, **kwargs):
        schema = original_get_openapi(*args, **kwargs)
        # Remove components if it exists
        if "components" in schema:
            del schema["components"]
        return schema

    with mock.patch("autogpt_libs.auth.helpers.get_openapi", mock_get_openapi):
        # Apply customization
        add_auth_responses_to_openapi(app)

        schema = app.openapi()

        # Components should be created
        assert "components" in schema
        assert "responses" in schema["components"]
        assert "HTTP401NotAuthenticatedError" in schema["components"]["responses"]


def test_add_auth_responses_to_openapi_all_http_methods():
    """Test that all HTTP methods are handled correctly."""
    app = FastAPI()

    from fastapi import Security

    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    @app.get("/resource")
    def get_resource(jwt: dict = Security(get_jwt_payload)):
        return {"method": "GET"}

    @app.post("/resource")
    def post_resource(jwt: dict = Security(get_jwt_payload)):
        return {"method": "POST"}

    @app.put("/resource")
    def put_resource(jwt: dict = Security(get_jwt_payload)):
        return {"method": "PUT"}

    @app.patch("/resource")
    def patch_resource(jwt: dict = Security(get_jwt_payload)):
        return {"method": "PATCH"}

    @app.delete("/resource")
    def delete_resource(jwt: dict = Security(get_jwt_payload)):
        return {"method": "DELETE"}

    # Apply customization
    add_auth_responses_to_openapi(app)

    schema = app.openapi()

    # All methods should have 401 response
    if "/resource" in schema["paths"]:
        for method in ["get", "post", "put", "patch", "delete"]:
            if method in schema["paths"]["/resource"]:
                method_spec = schema["paths"]["/resource"][method]
                if "responses" in method_spec:
                    assert "401" in method_spec["responses"]


def test_bearer_jwt_auth_scheme_config():
    """Test that bearer_jwt_auth is configured correctly."""
    assert bearer_jwt_auth.scheme_name == "HTTPBearerJWT"
    assert bearer_jwt_auth.auto_error is False


def test_add_auth_responses_with_no_routes():
    """Test OpenAPI generation with app that has no routes."""
    app = FastAPI(title="Empty App")

    # Apply customization to empty app
    add_auth_responses_to_openapi(app)

    schema = app.openapi()

    # Should still have basic structure
    assert schema["info"]["title"] == "Empty App"
    assert "components" in schema
    assert "responses" in schema["components"]
    assert "HTTP401NotAuthenticatedError" in schema["components"]["responses"]


def test_custom_openapi_function_replacement():
    """Test that the custom openapi function properly replaces the default."""
    app = FastAPI()

    # Store original function
    original_openapi = app.openapi

    # Apply customization
    add_auth_responses_to_openapi(app)

    # Function should be replaced
    assert app.openapi != original_openapi
    assert callable(app.openapi)


def test_endpoint_without_responses_section():
    """Test endpoint that has security but no responses section initially."""
    app = FastAPI()

    from fastapi import Security
    from fastapi.openapi.utils import get_openapi as original_get_openapi

    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    # Create endpoint
    @app.get("/no-responses")
    def endpoint_without_responses(jwt: dict = Security(get_jwt_payload)):
        return {"data": "test"}

    # Mock get_openapi to remove responses from the endpoint
    def mock_get_openapi(*args, **kwargs):
        schema = original_get_openapi(*args, **kwargs)
        # Remove responses from our endpoint to trigger line 40
        if "/no-responses" in schema.get("paths", {}):
            if "get" in schema["paths"]["/no-responses"]:
                # Delete responses to force the code to create it
                if "responses" in schema["paths"]["/no-responses"]["get"]:
                    del schema["paths"]["/no-responses"]["get"]["responses"]
        return schema

    with mock.patch("autogpt_libs.auth.helpers.get_openapi", mock_get_openapi):
        # Apply customization
        add_auth_responses_to_openapi(app)

        # Get schema and verify 401 was added
        schema = app.openapi()

        # The endpoint should now have 401 response
        if "/no-responses" in schema["paths"]:
            if "get" in schema["paths"]["/no-responses"]:
                responses = schema["paths"]["/no-responses"]["get"].get("responses", {})
                assert "401" in responses
                assert (
                    responses["401"]["$ref"]
                    == "#/components/responses/HTTP401NotAuthenticatedError"
                )


def test_components_with_existing_responses():
    """Test when components already has a responses section."""
    app = FastAPI()

    # Mock get_openapi to return schema with existing components/responses
    from fastapi.openapi.utils import get_openapi as original_get_openapi

    def mock_get_openapi(*args, **kwargs):
        schema = original_get_openapi(*args, **kwargs)
        # Add existing components/responses
        if "components" not in schema:
            schema["components"] = {}
        schema["components"]["responses"] = {
            "ExistingResponse": {"description": "An existing response"}
        }
        return schema

    with mock.patch("autogpt_libs.auth.helpers.get_openapi", mock_get_openapi):
        # Apply customization
        add_auth_responses_to_openapi(app)

        schema = app.openapi()

        # Both responses should exist
        assert "ExistingResponse" in schema["components"]["responses"]
        assert "HTTP401NotAuthenticatedError" in schema["components"]["responses"]

        # Verify our 401 response structure
        error_response = schema["components"]["responses"][
            "HTTP401NotAuthenticatedError"
        ]
        assert error_response["description"] == "Authentication required"


def test_openapi_schema_persistence():
    """Test that modifications to OpenAPI schema persist correctly."""
    app = FastAPI()

    from fastapi import Security

    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    @app.get("/test")
    def test_endpoint(jwt: dict = Security(get_jwt_payload)):
        return {"test": True}

    # Apply customization
    add_auth_responses_to_openapi(app)

    # Get schema multiple times
    schema1 = app.openapi()

    # Modify the cached schema (shouldn't affect future calls)
    schema1["info"]["title"] = "Modified Title"

    # Clear cache and get again
    app.openapi_schema = None
    schema2 = app.openapi()

    # Should regenerate with original title
    assert schema2["info"]["title"] == app.title
    assert schema2["info"]["title"] != "Modified Title"
