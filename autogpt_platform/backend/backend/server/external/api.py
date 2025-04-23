from fastapi import FastAPI

from .routes.v1 import v1_router

external_app = FastAPI(
    title="AutoGPT External API",
    description="External API for AutoGPT integrations",
    docs_url="/docs",
    version="1.0",
)
external_app.include_router(v1_router, prefix="/v1")
