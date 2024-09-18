import contextlib

import fastapi
import fastapi.responses
import fastapi.middleware.cors

import autogpt_libs.auth.middleware
import autogpt_server.data.block
import autogpt_server.data.db
import autogpt_server.data.graph
import autogpt_server.data.user
import autogpt_server.server.routes
import autogpt_server.server.utils
import autogpt_server.util.settings

from autogpt_server.data import user as user_db

settings = autogpt_server.util.settings.Settings()


@contextlib.asynccontextmanager
async def app_lifespan(app: fastapi.FastAPI):
    await autogpt_server.data.db.connect()
    await autogpt_server.data.block.initialize_blocks()

    if await user_db.create_default_user(settings.config.enable_auth):
        await autogpt_server.data.graph.import_packaged_templates()
    yield

    await autogpt_server.data.db.disconnect()


app = fastapi.FastAPI(
    title="AutoGPT Agent Server",
    description=(
        "This server is used to execute agents that are created by the "
        "AutoGPT system."
    ),
    summary="AutoGPT Agent Server",
    version="0.1",
    lifespan=app_lifespan,
)

api_router = fastapi.APIRouter(prefix="/api/v1")
api_router.dependencies.append(
    fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware)
)

api_router.include_router(autogpt_server.server.routes.root_router)
api_router.include_router(autogpt_server.server.routes.agents_router, tags=["agents"])
api_router.include_router(autogpt_server.server.routes.blocks_router, tags=["blocks"])
api_router.include_router(
    autogpt_server.server.routes.integrations_router, prefix="/integrations"
)

app.include_router(api_router)


@app.exception_handler(500)
def handle_internal_http_error(request: fastapi.Request, exc: Exception):
    return fastapi.responses.JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )


@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def validation_exception_handler(request: fastapi.Request, exc: fastapi.exceptions.RequestValidationError):
    errors = []
    for err in exc.errors():
        error = {
            "field": ".".join(err["loc"][1:]),  # Skipping 'body' or 'query' etc.
            "message": err["msg"],
            "type": err["type"]
        }
        errors.append(error)
    
    return fastapi.responses.JSONResponse(
        status_code=422,
        content={
            "status": "fail",
            "errors": errors
        },
    )

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
