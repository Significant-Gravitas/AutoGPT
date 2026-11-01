"""
External API Application

This module defines the FastAPI application mounted at /external-api, which
serves v1. The host app mounts v2 at /api/v2 itself.
"""

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from backend.monitoring.instrumentation import instrument_fastapi

from .v1.app import v1_app

# Each version publishes its own spec and docs; this app only routes to them.
external_api = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@external_api.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root to the current version's documentation."""
    # Relative, so it keeps a proxy prefix such as the single container's /_agpt.
    return RedirectResponse(url="../api/v2/docs")


@external_api.get("/openapi.json", include_in_schema=False)
async def v1_spec_redirect(request: Request) -> RedirectResponse:
    """v1's spec, at the address it had before the API was versioned, where
    published docs and generated clients still fetch it."""
    return RedirectResponse(
        url=f"{request.scope.get('root_path', '')}/v1/openapi.json", status_code=308
    )


external_api.mount("/v1", v1_app)

# The REST API process this app is mounted in serves the metrics on their own
# port, never on this public app.
instrument_fastapi(external_api, service_name="external-api", expose_endpoint=False)
