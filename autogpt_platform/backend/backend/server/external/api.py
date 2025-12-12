from fastapi import FastAPI

from backend.monitoring.instrumentation import instrument_fastapi
from backend.server.middleware.security import SecurityHeadersMiddleware

from .routes.execution import execution_router
from .routes.grants import grants_router

external_app = FastAPI(
    title="AutoGPT External API",
    description="External API for AutoGPT integrations (OAuth-based)",
    docs_url="/docs",
    version="1.0",
)

external_app.add_middleware(SecurityHeadersMiddleware)
external_app.include_router(grants_router, prefix="/v1")
external_app.include_router(execution_router, prefix="/v1")

# Add Prometheus instrumentation
instrument_fastapi(
    external_app,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=True,
)
