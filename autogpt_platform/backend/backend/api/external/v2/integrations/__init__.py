"""
V2 External API - Integrations Package

Aggregates all integration-related sub-routers.
"""

from fastapi import APIRouter

from .credentials import credentials_router

integrations_router = APIRouter(tags=["integrations"])

integrations_router.include_router(credentials_router)
