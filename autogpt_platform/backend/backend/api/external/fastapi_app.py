"""
External API Application

This module defines the main FastAPI application for the external API,
which mounts the v1 and v2 sub-applications.
"""

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from backend.monitoring.instrumentation import instrument_fastapi
from backend.util.settings import AppEnvironment, Settings

from .v1.app import v1_app
from .v2.app import v2_app

settings = Settings()

DESCRIPTION = """
The external API provides programmatic access to the AutoGPT Platform for building
integrations, automations, and custom applications.

### API Versions

| Version             | End of Life | Path                   | Documentation |
|---------------------|-------------|------------------------|---------------|
| **v2**              |             | `/external-api/v2/...` | [v2 docs](v2/docs) |
| **v1** (deprecated) | 2026-12-31  | `/external-api/v1/...` | [v1 docs](v1/docs) |

**Recommendation**: New integrations should use v2.

For authentication details and usage examples, see the
[API Integration Guide](https://agpt.co/docs/platform/api-and-integrations/api-guide).
"""

external_api = FastAPI(
    title="AutoGPT Platform API",
    summary="External API for AutoGPT Platform integrations",
    description=DESCRIPTION,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    # `/openapi.json` was v1's spec before v2 existed, and published docs and
    # generated clients still fetch it there; see `v1_spec_redirect`.
    openapi_url="/versions.json",
)


@external_api.get("/", include_in_schema=False)
async def root_redirect(request: Request) -> RedirectResponse:
    """Redirect root to this API's documentation, not the host app's."""
    return RedirectResponse(url=f"{request.scope.get('root_path', '')}/docs")


@external_api.get("/openapi.json", include_in_schema=False)
async def v1_spec_redirect(request: Request) -> RedirectResponse:
    """v1's spec, at the address it had before the API was versioned."""
    return RedirectResponse(
        url=f"{request.scope.get('root_path', '')}/v1/openapi.json", status_code=308
    )


# Mount versioned sub-applications
# Each sub-app has its own /docs page at /v1/docs and /v2/docs
external_api.mount("/v1", v1_app)
external_api.mount("/v2", v2_app)

# The scrape endpoint stays mounted for Prometheus everywhere; only the local
# docs advertise it, which is how the internal app has treated it since it landed.
instrument_fastapi(
    external_api,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=settings.config.app_env == AppEnvironment.LOCAL,
)
