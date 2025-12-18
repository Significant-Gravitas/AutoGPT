from fastapi import FastAPI

from backend.monitoring.instrumentation import instrument_fastapi
from backend.server.middleware.security import SecurityHeadersMiddleware

from .routes.integrations import integrations_router
from .routes.tools import tools_router
from .routes.v1 import v1_router

external_app = FastAPI(
    title="AutoGPT External API",
    description="External API for AutoGPT integrations",
    docs_url="/docs",
    version="1.0",
)

external_app.add_middleware(SecurityHeadersMiddleware)
external_app.include_router(v1_router, prefix="/v1")
external_app.include_router(tools_router, prefix="/v1")
external_app.include_router(integrations_router, prefix="/v1")

# Add Prometheus instrumentation
instrument_fastapi(
    external_app,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=True,
)
