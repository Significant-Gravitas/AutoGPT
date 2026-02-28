"""
V2 External API Application

This module defines the FastAPI application for the v2 external API.
"""

from fastapi import FastAPI

from backend.api.external.middleware import add_auth_responses_to_openapi
from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.api.utils.openapi import sort_openapi

from .routes import v2_router

DESCRIPTION = """
The v2 API provides comprehensive access to the AutoGPT Platform for building
integrations, automations, and custom applications.

### Key Improvements over v1

- **Consistent naming**: Uses `graph_id`/`graph_version` consistently
- **Better pagination**: All list endpoints support pagination
- **Comprehensive coverage**: Access to library, runs, schedules, credits, and more
- **Human-in-the-loop**: Review and approve agent decisions via the API

For authentication details and usage examples, see the
[API Integration Guide](https://docs.agpt.co/platform/integrating/api-guide/).

### Pagination

List endpoints return paginated responses. Use `page` and `page_size` query
parameters to navigate results. Maximum page size is 100 items.
"""

v2_app = FastAPI(
    title="AutoGPT Platform External API",
    summary="External API for AutoGPT Platform integrations (v2)",
    description=DESCRIPTION,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "graphs",
            "description": "Create, update, and manage agent graphs",
        },
        {
            "name": "schedules",
            "description": "Manage scheduled graph executions",
        },
        {
            "name": "blocks",
            "description": "Discover available building blocks",
        },
        {
            "name": "marketplace",
            "description": "Browse agents and creators, manage submissions",
        },
        {
            "name": "library",
            "description": "Access your agent library and execute agents",
        },
        {
            "name": "runs",
            "description": "Monitor execution runs and human-in-the-loop reviews",
        },
        {
            "name": "credits",
            "description": "Check balance and view transaction history",
        },
        {
            "name": "integrations",
            "description": "Manage OAuth credentials for external services",
        },
        {
            "name": "files",
            "description": "Upload files for agent input",
        },
    ],
)

v2_app.add_middleware(SecurityHeadersMiddleware)
v2_app.include_router(v2_router)

# Add 401 responses to authenticated endpoints in OpenAPI spec
add_auth_responses_to_openapi(v2_app)
# Sort OpenAPI schema to eliminate diff on refactors
sort_openapi(v2_app)
