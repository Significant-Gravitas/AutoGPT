"""
External API Application

This module defines the main FastAPI application for the external API,
which mounts the v1 and v2 sub-applications.
"""

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from backend.monitoring.instrumentation import instrument_fastapi

from .v1.app import v1_app
from .v2.app import v2_app

# Each version publishes its own spec and docs; this app only routes to them.
external_api = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@external_api.get("/", include_in_schema=False)
async def root_redirect(request: Request) -> RedirectResponse:
    """Redirect root to the current version's documentation."""
    return RedirectResponse(url=f"{request.scope.get('root_path', '')}/v2/docs")


@external_api.get("/openapi.json", include_in_schema=False)
async def v1_spec_redirect(request: Request) -> RedirectResponse:
    """v1's spec, at the address it had before the API was versioned, where
    published docs and generated clients still fetch it."""
    return RedirectResponse(
        url=f"{request.scope.get('root_path', '')}/v1/openapi.json", status_code=308
    )


# Mount versioned sub-applications
# Each sub-app has its own /docs page at /v1/docs and /v2/docs
external_api.mount("/v1", v1_app)
external_api.mount("/v2", v2_app)

# The scrape endpoint stays mounted for Prometheus everywhere.
instrument_fastapi(
    external_api,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
)
