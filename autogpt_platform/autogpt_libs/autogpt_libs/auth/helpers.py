from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from .jwt_utils import bearer_jwt_auth


def add_auth_responses_to_openapi(app: FastAPI) -> None:
    """
    Set up custom OpenAPI schema generation that adds 401 responses
    to all authenticated endpoints.

    This is needed when using HTTPBearer with auto_error=False to get proper
    401 responses instead of 403, but FastAPI only automatically adds security
    responses when auto_error=True.
    """

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add 401 response to all endpoints that have security requirements
        for path, methods in openapi_schema["paths"].items():
            for method, details in methods.items():
                security_schemas = [
                    schema
                    for auth_option in details.get("security", [])
                    for schema in auth_option.keys()
                ]
                if bearer_jwt_auth.scheme_name not in security_schemas:
                    continue

                if "responses" not in details:
                    details["responses"] = {}

                details["responses"]["401"] = {
                    "$ref": "#/components/responses/HTTP401NotAuthenticatedError"
                }

        # Ensure #/components/responses exists
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}
        if "responses" not in openapi_schema["components"]:
            openapi_schema["components"]["responses"] = {}

        # Define 401 response
        openapi_schema["components"]["responses"]["HTTP401NotAuthenticatedError"] = {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"detail": {"type": "string"}},
                    }
                }
            },
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi
