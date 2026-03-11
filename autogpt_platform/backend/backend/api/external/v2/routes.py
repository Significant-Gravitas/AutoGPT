"""
V2 External API Routes

This module defines the main v2 router that aggregates all v2 API endpoints.
"""

from fastapi import APIRouter

from .blocks import blocks_router
from .credits import credits_router
from .files import file_workspace_router
from .graphs import graphs_router
from .integrations import integrations_router
from .library import library_router
from .marketplace import marketplace_router
from .runs import runs_router
from .schedules import graph_schedules_router, schedules_router
from .search import search_router

v2_router = APIRouter()

# Include all sub-routers
v2_router.include_router(blocks_router, prefix="/blocks")
v2_router.include_router(credits_router, prefix="/credits")
v2_router.include_router(file_workspace_router, prefix="/files")
v2_router.include_router(graph_schedules_router, prefix="/graphs")
v2_router.include_router(graphs_router, prefix="/graphs")
v2_router.include_router(integrations_router, prefix="/integrations")
v2_router.include_router(library_router, prefix="/library")
v2_router.include_router(marketplace_router, prefix="/marketplace")
v2_router.include_router(runs_router, prefix="/runs")
v2_router.include_router(schedules_router, prefix="/schedules")
v2_router.include_router(search_router, prefix="/search")
