"""
V2 External API Routes

This module defines the main v2 router that aggregates all v2 API endpoints.
"""

from fastapi import APIRouter

from .blocks import blocks_router
from .credits import credits_router
from .files import files_router
from .graphs import graphs_router
from .integrations import integrations_router
from .library import library_router
from .marketplace import marketplace_router
from .runs import runs_router
from .schedules import graph_schedules_router, schedules_router

v2_router = APIRouter()

# Include all sub-routers
v2_router.include_router(graphs_router, prefix="/graphs", tags=["graphs"])
v2_router.include_router(graph_schedules_router, prefix="/graphs", tags=["schedules"])
v2_router.include_router(schedules_router, prefix="/schedules", tags=["schedules"])
v2_router.include_router(blocks_router, prefix="/blocks", tags=["blocks"])
v2_router.include_router(
    marketplace_router, prefix="/marketplace", tags=["marketplace"]
)
v2_router.include_router(library_router, prefix="/library", tags=["library"])
v2_router.include_router(runs_router, prefix="/runs", tags=["runs"])
v2_router.include_router(credits_router, prefix="/credits", tags=["credits"])
v2_router.include_router(
    integrations_router, prefix="/integrations", tags=["integrations"]
)
v2_router.include_router(files_router, prefix="/files", tags=["files"])
