"""
V2 External API Application

This module defines the FastAPI application for the v2 external API.
"""

from fastapi import FastAPI

from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.api.utils.openapi import sort_openapi

from .errors import add_v2_exception_handlers
from .global_rate_limit import GlobalRateLimitMiddleware
from .mcp_server import create_mcp_app
from .routes import v2_router

DESCRIPTION = """
The v2 API provides comprehensive access to the AutoGPT Platform for building
integrations, automations, and custom applications.

### Key Improvements over v1

- **Consistent naming**: Uses `graph_id`/`graph_version` consistently
- **One list shape**: every list endpoint takes `limit`/`cursor` and returns
  `{"items": [...], "next_cursor": ..., "total_count": ...}`
- **One error shape**: every non-2xx response is
  `{"error": {"code", "message", "details"}}`
- **Comprehensive coverage**: Access to library, runs, schedules, credits, and more
- **Human-in-the-loop**: Review and approve agent decisions via the API

For authentication details and usage examples, see the
[API Integration Guide](https://docs.agpt.co/platform/integrating/api-guide/).

### Rate Limits

All endpoints are subject to a global rate limit per authenticated user:
**200 requests per minute**. Unauthenticated requests are limited to
**5 requests per minute** per client IP.

Some endpoints have additional per-endpoint rate limits (documented on each
endpoint). When a rate limit is exceeded, the API returns `429 Too Many Requests`.

### Pagination

List endpoints take `limit` (1-100, default 20) and `cursor`, and return
`{"items": [...], "next_cursor": ..., "total_count": ...}`. Pass `next_cursor`
back as `cursor` for the next page; it is `null` on the last page. Cursors are
opaque. `total_count` counts the matches across all pages, and is `null` only
where the source cannot report one.

### Status codes

`201` when a resource is created, `202` when work is queued, `204` on delete,
`200` on everything else.

### Errors

Every non-2xx response is `{"error": {"code", "message", "details"}}`. Branch
on `code`, a stable snake_case identifier; `message` is for humans.
""".strip()

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
                "Monitor, stop, delete, and share agent runs; "
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

v2_app.add_middleware(GlobalRateLimitMiddleware)
v2_app.add_middleware(SecurityHeadersMiddleware)
v2_app.include_router(v2_router)

# Mounted sub-apps do NOT inherit exception handlers from the parent app,
# so we must register them here for the v2 API specifically.
add_v2_exception_handlers(v2_app)

# Mount MCP server (Copilot tools via Streamable HTTP)
v2_app.mount("/mcp", create_mcp_app())

# Sort OpenAPI schema to eliminate diff on refactors
sort_openapi(v2_app)
