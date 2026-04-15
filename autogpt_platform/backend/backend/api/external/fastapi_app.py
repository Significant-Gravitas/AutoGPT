from fastapi import FastAPI

from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.monitoring.instrumentation import instrument_fastapi

from .v1.routes import v1_router

external_api = FastAPI(
    title="AutoGPT External API",
    description="External API for AutoGPT integrations",
    docs_url="/docs",
    version="1.0",
)

external_api.add_middleware(SecurityHeadersMiddleware)
external_api.include_router(v1_router, prefix="/v1")

# Add Prometheus instrumentation
instrument_fastapi(
    external_api,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=True,
)
