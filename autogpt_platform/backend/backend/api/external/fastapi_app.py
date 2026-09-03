"""
External API Application

This module defines the main FastAPI application for the external API,
which mounts the v1 and v2 sub-applications.
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from backend.monitoring.instrumentation import instrument_fastapi

from .v1.app import v1_app
from .v2.app import v2_app

DESCRIPTION = """
The external API provides programmatic access to the AutoGPT Platform for building
integrations, automations, and custom applications.

### API Versions

| Version             | End of Life | Path                   | Documentation |
|---------------------|-------------|------------------------|---------------|
| **v2**              |             | `/external-api/v2/...` | [v2 docs](v2/docs) |
| **v1** (deprecated) | 2025-05-01  | `/external-api/v1/...` | [v1 docs](v1/docs) |

**Recommendation**: New integrations should use v2.

For authentication details and usage examples, see the
[API Integration Guide](https://docs.agpt.co/platform/integrating/api-guide/).
"""

external_api = FastAPI(
    title="AutoGPT Platform API",
    summary="External API for AutoGPT Platform integrations",
    description=DESCRIPTION,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@external_api.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# Mount versioned sub-applications
# Each sub-app has its own /docs page at /v1/docs and /v2/docs
external_api.mount("/v1", v1_app)
external_api.mount("/v2", v2_app)

# Add Prometheus instrumentation to the main app
instrument_fastapi(
    external_api,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=True,
)
