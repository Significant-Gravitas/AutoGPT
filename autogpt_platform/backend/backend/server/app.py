import contextlib

import fastapi
import fastapi.middleware.cors
import fastapi.responses

import backend.data.block
import backend.data.db
import backend.data.user
import backend.server.routers.v1
import backend.util.settings

settings = backend.util.settings.Settings()


@contextlib.asynccontextmanager
async def lifespan_context(app: fastapi.FastAPI):
    await backend.data.db.connect()
    await backend.data.block.initialize_blocks()
    await backend.data.user.migrate_and_encrypt_user_integrations()
    yield
    await backend.data.db.disconnect()


docs_url = (
    "/docs"
    if settings.config.app_env == backend.util.settings.AppEnvironment.LOCAL
    else None
)

app = fastapi.FastAPI(
    title="AutoGPT Agent Server",
    description=(
        "This server is used to execute agents that are created by the "
        "AutoGPT system."
    ),
    summary="AutoGPT Agent Server",
    version="0.1",
    lifespan=lifespan_context,
    docs_url=docs_url,
)

app.include_router(backend.server.routers.v1.v1_router)
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=settings.config.backend_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get(path="/health", tags=["health"], dependencies=[])
async def health():
    return {"status": "healthy"}


@app.exception_handler(Exception)
def handle_internal_http_error(request: fastapi.Request, exc: Exception):
    return fastapi.responses.JSONResponse(
        content={
            "message": f"{request.method} {request.url.path} failed",
            "error": str(exc),
        },
        status_code=500,
    )
