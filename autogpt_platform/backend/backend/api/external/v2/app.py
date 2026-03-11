"""
V2 External API Application

This module defines the FastAPI application for the v2 external API.
"""

from fastapi import FastAPI

from backend.api.external.middleware import add_auth_responses_to_openapi
from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.api.utils.exceptions import add_exception_handlers
from backend.api.utils.openapi import sort_openapi

from .mcp_server import create_mcp_app
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
            "name": "search",
            "description": "Cross-domain hybrid search across agents, blocks, and docs",
        },
        {
            "name": "marketplace",
            "description": "Browse agents and creators, manage submissions",
        },
        {
            "name": "library",
            "description": (
                "Manage your agent library (agents and presets), "
                "execute agents, organize with folders"
            ),
        },
        {
            "name": "presets",
            "description": "Agent execution presets with webhook triggers",
        },
        {
            "name": "runs",
            "description": (
                "Monitor, stop, delete, and share execution runs; "
                "manage human-in-the-loop reviews"
            ),
        },
        {
            "name": "credits",
            "description": "Check balance and view transaction history",
        },
        {
            "name": "integrations",
            "description": "List, create, and delete integration credentials",
        },
        {
            "name": "files",
            "description": "Upload, list, download, and delete workspace files",
        },
    ],
)

v2_app.add_middleware(SecurityHeadersMiddleware)
v2_app.include_router(v2_router)

# Mounted sub-apps do NOT inherit exception handlers from the parent app,
# so we must register them here for the v2 API specifically.
add_exception_handlers(v2_app)

# Mount MCP server (Copilot tools via Streamable HTTP)
v2_app.mount("/mcp", create_mcp_app())

# Add 401 responses to authenticated endpoints in OpenAPI spec
add_auth_responses_to_openapi(v2_app)
# Sort OpenAPI schema to eliminate diff on refactors
sort_openapi(v2_app)
