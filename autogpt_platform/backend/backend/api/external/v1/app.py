"""
V1 External API Application

This module defines the FastAPI application for the v1 external API.
"""

from fastapi import FastAPI

from backend.api.external.middleware import add_auth_responses_to_openapi
from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.api.utils.openapi import sort_openapi

from .routes import v1_router

DESCRIPTION = """
The v1 API provides access to core AutoGPT functionality for external integrations.

For authentication details and usage examples, see the
[API Integration Guide](https://docs.agpt.co/platform/integrating/api-guide/).
"""

v1_app = FastAPI(
    title="AutoGPT Platform API",
    summary="External API for AutoGPT Platform integrations (v1)",
    description=DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "user", "description": "User information"},
        {"name": "blocks", "description": "Block operations"},
        {"name": "graphs", "description": "Graph execution"},
        {"name": "store", "description": "Marketplace agents and creators"},
        {"name": "integrations", "description": "OAuth credential management"},
        {"name": "tools", "description": "AI assistant tools"},
    ],
)

v1_app.add_middleware(SecurityHeadersMiddleware)
v1_app.include_router(v1_router)

# Add 401 responses to authenticated endpoints in OpenAPI spec
add_auth_responses_to_openapi(v1_app)
# Sort OpenAPI schema to eliminate diff on refactors
sort_openapi(v1_app)
