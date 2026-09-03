"""
V2 External API Routes

This module defines the main v2 router that aggregates all v2 API endpoints.
"""

from fastapi import APIRouter
from starlette import status

from .blocks import blocks_router
from .credits import credits_router
from .errors import ErrorResponse
from .files import file_workspace_router
from .graphs import graphs_router
from .identity import identity_router
from .integrations import integrations_router
from .library import library_router
from .marketplace import marketplace_router
from .runs import runs_router
from .schedules import graph_schedules_router, schedules_router
from .search import search_router

# Declared once, on the parent router, so every v2 operation advertises the one
# error body shape instead of FastAPI's default `{"detail": ...}`.
ERROR_RESPONSES: dict[int | str, dict] = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        status.HTTP_400_BAD_REQUEST: "Malformed request",
        status.HTTP_401_UNAUTHORIZED: "Missing or invalid credentials",
        status.HTTP_403_FORBIDDEN: "Credentials lack a required scope",
        status.HTTP_404_NOT_FOUND: "No such resource",
        status.HTTP_422_UNPROCESSABLE_CONTENT: "Request failed validation",
        status.HTTP_429_TOO_MANY_REQUESTS: "Rate limit exceeded",
        status.HTTP_500_INTERNAL_SERVER_ERROR: "Unhandled server error",
    }.items()
}

v2_router = APIRouter(responses=ERROR_RESPONSES)

# Include all sub-routers
v2_router.include_router(blocks_router, prefix="/blocks")
v2_router.include_router(credits_router, prefix="/credits")
v2_router.include_router(file_workspace_router, prefix="/files")
v2_router.include_router(graph_schedules_router, prefix="/graphs")
v2_router.include_router(graphs_router, prefix="/graphs")
v2_router.include_router(identity_router)
v2_router.include_router(integrations_router, prefix="/integrations")
v2_router.include_router(library_router, prefix="/library")
v2_router.include_router(marketplace_router, prefix="/marketplace")
v2_router.include_router(runs_router, prefix="/runs")
v2_router.include_router(schedules_router, prefix="/schedules")
v2_router.include_router(search_router, prefix="/search")
