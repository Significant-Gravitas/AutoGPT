"""
V2 External API - Library Package

Aggregates all library-related sub-routers (agents, folders).
"""

from fastapi import APIRouter

from .agents import agents_router
from .folders import folders_router

library_router = APIRouter()

library_router.include_router(agents_router)
library_router.include_router(folders_router)
