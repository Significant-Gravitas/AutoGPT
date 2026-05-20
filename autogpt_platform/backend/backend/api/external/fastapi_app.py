from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from backend.api.middleware.security import SecurityHeadersMiddleware
from backend.copilot.rate_limit import UserPaywalledError
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


# UserPaywalledError raised at deep enqueue paths (e.g. add_graph_execution)
# maps to HTTP 402 here too — the external API is a separate FastAPI app
# so the handler in rest_api.py doesn't reach it.
@external_api.exception_handler(UserPaywalledError)
async def _user_paywalled_handler(_request: Request, exc: UserPaywalledError):
    return JSONResponse(status_code=402, content={"detail": str(exc)})


# Add Prometheus instrumentation
instrument_fastapi(
    external_api,
    service_name="external-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=True,
)
