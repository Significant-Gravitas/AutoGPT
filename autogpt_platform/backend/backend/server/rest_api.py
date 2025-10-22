import contextlib
import logging
import platform
from enum import Enum
from typing import Any, Optional

import fastapi
import fastapi.responses
import pydantic
import starlette.middleware.cors
import uvicorn
from autogpt_libs.auth import add_auth_responses_to_openapi
from autogpt_libs.auth import verify_settings as verify_auth_settings
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.routing import APIRoute
from prisma.errors import PrismaError

import backend.data.block
import backend.data.db
import backend.data.graph
import backend.data.user
import backend.integrations.webhooks.utils
import backend.server.routers.postmark.postmark
import backend.server.routers.v1
import backend.server.v2.admin.credit_admin_routes
import backend.server.v2.admin.store_admin_routes
import backend.server.v2.builder
import backend.server.v2.builder.routes
import backend.server.v2.library.db
import backend.server.v2.library.model
import backend.server.v2.library.routes
import backend.server.v2.otto.routes
import backend.server.v2.store.model
import backend.server.v2.store.routes
import backend.server.v2.turnstile.routes
import backend.util.service
import backend.util.settings
from backend.blocks.llm import LlmModel
from backend.data.model import Credentials
from backend.integrations.providers import ProviderName
from backend.monitoring.instrumentation import instrument_fastapi
from backend.server.external.api import external_app
from backend.server.middleware.security import SecurityHeadersMiddleware
from backend.util import json
from backend.util.cloud_storage import shutdown_cloud_storage_handler
from backend.util.exceptions import NotAuthorizedError, NotFoundError
from backend.util.feature_flag import initialize_launchdarkly, shutdown_launchdarkly
from backend.util.service import UnhealthyServiceError

settings = backend.util.settings.Settings()
logger = logging.getLogger(__name__)

logging.getLogger("autogpt_libs").setLevel(logging.INFO)


@contextlib.contextmanager
def launch_darkly_context():
    if settings.config.app_env != backend.util.settings.AppEnvironment.LOCAL:
        initialize_launchdarkly()
        try:
            yield
        finally:
            shutdown_launchdarkly()
    else:
        yield


@contextlib.asynccontextmanager
async def lifespan_context(app: fastapi.FastAPI):
    verify_auth_settings()

    await backend.data.db.connect()

    # Configure thread pool for FastAPI sync operation performance
    # CRITICAL: FastAPI automatically runs ALL sync functions in this thread pool:
    # - Any endpoint defined with 'def' (not async def)
    # - Any dependency function defined with 'def' (not async def)
    # - Manual run_in_threadpool() calls (like JWT decoding)
    # Default pool size is only 40 threads, causing bottlenecks under high concurrency
    config = backend.util.settings.Config()
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            config.fastapi_thread_pool_size
        )
        logger.info(
            f"Thread pool size set to {config.fastapi_thread_pool_size} for sync endpoint/dependency performance"
        )
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not configure thread pool size: {e}")
        # Continue without thread pool configuration

    # Ensure SDK auto-registration is patched before initializing blocks
    from backend.sdk.registry import AutoRegistry

    AutoRegistry.patch_integrations()

    await backend.data.block.initialize_blocks()

    await backend.data.user.migrate_and_encrypt_user_integrations()
    await backend.data.graph.fix_llm_provider_credentials()
    await backend.data.graph.migrate_llm_models(LlmModel.GPT4O)
    await backend.integrations.webhooks.utils.migrate_legacy_triggered_graphs()

    with launch_darkly_context():
        yield

    try:
        await shutdown_cloud_storage_handler()
    except Exception as e:
        logger.warning(f"Error shutting down cloud storage handler: {e}")

    await backend.data.db.disconnect()


def custom_generate_unique_id(route: APIRoute):
    """Generate clean operation IDs for OpenAPI spec following the format:
    {method}{tag}{summary}
    """
    if not route.tags or not route.methods:
        return f"{route.name}"

    method = list(route.methods)[0].lower()
    first_tag = route.tags[0]
    if isinstance(first_tag, Enum):
        tag_str = first_tag.name
    else:
        tag_str = str(first_tag)

    tag = "".join(word.capitalize() for word in tag_str.split("_"))  # v1/v2

    summary = (
        route.summary if route.summary else route.name
    )  # need to be unique, a different version could have the same summary
    summary = "".join(word.capitalize() for word in str(summary).split("_"))

    if tag:
        return f"{method}{tag}{summary}"
    else:
        return f"{method}{summary}"


docs_url = (
    "/docs"
    if settings.config.app_env == backend.util.settings.AppEnvironment.LOCAL
    else None
)

app = fastapi.FastAPI(
    title="AutoGPT Agent Server",
    description=(
        "This server is used to execute agents that are created by the AutoGPT system."
    ),
    summary="AutoGPT Agent Server",
    version="0.1",
    lifespan=lifespan_context,
    docs_url=docs_url,
    generate_unique_id_function=custom_generate_unique_id,
)

app.add_middleware(SecurityHeadersMiddleware)

# Add GZip compression middleware for large responses (like /api/blocks)
app.add_middleware(GZipMiddleware, minimum_size=50_000)  # 50KB threshold

# Add 401 responses to authenticated endpoints in OpenAPI spec
add_auth_responses_to_openapi(app)

# Add Prometheus instrumentation
instrument_fastapi(
    app,
    service_name="rest-api",
    expose_endpoint=True,
    endpoint="/metrics",
    include_in_schema=settings.config.app_env
    == backend.util.settings.AppEnvironment.LOCAL,
)


def handle_internal_http_error(status_code: int = 500, log_error: bool = True):
    def handler(request: fastapi.Request, exc: Exception):
        if log_error:
            logger.exception(
                "%s %s failed. Investigate and resolve the underlying issue: %s",
                request.method,
                request.url.path,
                exc,
            )

        hint = (
            "Adjust the request and retry."
            if status_code < 500
            else "Check server logs and dependent services."
        )
        return fastapi.responses.JSONResponse(
            content={
                "message": f"Failed to process {request.method} {request.url.path}",
                "detail": str(exc),
                "hint": hint,
            },
            status_code=status_code,
        )

    return handler


async def validation_error_handler(
    request: fastapi.Request, exc: Exception
) -> fastapi.responses.Response:
    logger.error(
        "Validation failed for %s %s: %s. Fix the request payload and try again.",
        request.method,
        request.url.path,
        exc,
    )
    errors: list | str
    if hasattr(exc, "errors"):
        errors = exc.errors()  # type: ignore[call-arg]
    else:
        errors = str(exc)

    response_content = {
        "message": f"Invalid data for {request.method} {request.url.path}",
        "detail": errors,
        "hint": "Ensure the request matches the API schema.",
    }

    content_json = json.dumps(response_content)

    return fastapi.responses.Response(
        content=content_json,
        status_code=422,
        media_type="application/json",
    )


app.add_exception_handler(PrismaError, handle_internal_http_error(500))
app.add_exception_handler(NotFoundError, handle_internal_http_error(404, False))
app.add_exception_handler(NotAuthorizedError, handle_internal_http_error(403, False))
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(pydantic.ValidationError, validation_error_handler)
app.add_exception_handler(ValueError, handle_internal_http_error(400))
app.add_exception_handler(Exception, handle_internal_http_error(500))

app.include_router(backend.server.routers.v1.v1_router, tags=["v1"], prefix="/api")
app.include_router(
    backend.server.v2.store.routes.router, tags=["v2"], prefix="/api/store"
)
app.include_router(
    backend.server.v2.builder.routes.router, tags=["v2"], prefix="/api/builder"
)
app.include_router(
    backend.server.v2.admin.store_admin_routes.router,
    tags=["v2", "admin"],
    prefix="/api/store",
)
app.include_router(
    backend.server.v2.admin.credit_admin_routes.router,
    tags=["v2", "admin"],
    prefix="/api/credits",
)
app.include_router(
    backend.server.v2.library.routes.router, tags=["v2"], prefix="/api/library"
)
app.include_router(
    backend.server.v2.otto.routes.router, tags=["v2", "otto"], prefix="/api/otto"
)
app.include_router(
    backend.server.v2.turnstile.routes.router,
    tags=["v2", "turnstile"],
    prefix="/api/turnstile",
)

app.include_router(
    backend.server.routers.postmark.postmark.router,
    tags=["v1", "email"],
    prefix="/api/email",
)

app.mount("/external-api", external_app)


@app.get(path="/health", tags=["health"], dependencies=[])
async def health():
    if not backend.data.db.is_connected():
        raise UnhealthyServiceError("Database is not connected")
    return {"status": "healthy"}


class AgentServer(backend.util.service.AppProcess):
    def run(self):
        server_app = starlette.middleware.cors.CORSMiddleware(
            app=app,
            allow_origins=settings.config.backend_cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )
        config = backend.util.settings.Config()

        # Configure uvicorn with performance optimizations from Kludex FastAPI tips
        uvicorn_config = {
            "app": server_app,
            "host": config.agent_api_host,
            "port": config.agent_api_port,
            "log_config": None,
            # Use httptools for HTTP parsing (if available)
            "http": "httptools",
            # Only use uvloop on Unix-like systems (not supported on Windows)
            "loop": "uvloop" if platform.system() != "Windows" else "auto",
        }

        # Only add debug in local environment (not supported in all uvicorn versions)
        if config.app_env == backend.util.settings.AppEnvironment.LOCAL:
            import os

            # Enable asyncio debug mode via environment variable
            os.environ["PYTHONASYNCIODEBUG"] = "1"

        uvicorn.run(**uvicorn_config)

    def cleanup(self):
        super().cleanup()
        logger.info(f"[{self.service_name}] ⏳ Shutting down Agent Server...")

    @staticmethod
    async def test_execute_graph(
        graph_id: str,
        user_id: str,
        graph_version: Optional[int] = None,
        node_input: Optional[dict[str, Any]] = None,
    ):
        return await backend.server.routers.v1.execute_graph(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            inputs=node_input or {},
            credentials_inputs={},
        )

    @staticmethod
    async def test_get_graph(
        graph_id: str,
        graph_version: int,
        user_id: str,
        for_export: bool = False,
    ):
        return await backend.server.routers.v1.get_graph(
            graph_id, user_id, graph_version, for_export
        )

    @staticmethod
    async def test_create_graph(
        create_graph: backend.server.routers.v1.CreateGraph,
        user_id: str,
    ):
        return await backend.server.routers.v1.create_new_graph(create_graph, user_id)

    @staticmethod
    async def test_get_graph_run_status(graph_exec_id: str, user_id: str):
        from backend.data.execution import get_graph_execution_meta

        execution = await get_graph_execution_meta(
            user_id=user_id, execution_id=graph_exec_id
        )
        if not execution:
            raise ValueError(f"Execution {graph_exec_id} not found")
        return execution.status

    @staticmethod
    async def test_delete_graph(graph_id: str, user_id: str):
        """Used for clean-up after a test run"""
        await backend.server.v2.library.db.delete_library_agent_by_graph_id(
            graph_id=graph_id, user_id=user_id
        )
        return await backend.server.routers.v1.delete_graph(graph_id, user_id)

    @staticmethod
    async def test_get_presets(user_id: str, page: int = 1, page_size: int = 10):
        return await backend.server.v2.library.routes.presets.list_presets(
            user_id=user_id, page=page, page_size=page_size
        )

    @staticmethod
    async def test_get_preset(preset_id: str, user_id: str):
        return await backend.server.v2.library.routes.presets.get_preset(
            preset_id=preset_id, user_id=user_id
        )

    @staticmethod
    async def test_create_preset(
        preset: backend.server.v2.library.model.LibraryAgentPresetCreatable,
        user_id: str,
    ):
        return await backend.server.v2.library.routes.presets.create_preset(
            preset=preset, user_id=user_id
        )

    @staticmethod
    async def test_update_preset(
        preset_id: str,
        preset: backend.server.v2.library.model.LibraryAgentPresetUpdatable,
        user_id: str,
    ):
        return await backend.server.v2.library.routes.presets.update_preset(
            preset_id=preset_id, preset=preset, user_id=user_id
        )

    @staticmethod
    async def test_delete_preset(preset_id: str, user_id: str):
        return await backend.server.v2.library.routes.presets.delete_preset(
            preset_id=preset_id, user_id=user_id
        )

    @staticmethod
    async def test_execute_preset(
        preset_id: str,
        user_id: str,
        inputs: Optional[dict[str, Any]] = None,
    ):
        return await backend.server.v2.library.routes.presets.execute_preset(
            preset_id=preset_id,
            user_id=user_id,
            inputs=inputs or {},
            credential_inputs={},
        )

    @staticmethod
    async def test_create_store_listing(
        request: backend.server.v2.store.model.StoreSubmissionRequest, user_id: str
    ):
        return await backend.server.v2.store.routes.create_submission(request, user_id)

    ### ADMIN ###

    @staticmethod
    async def test_review_store_listing(
        request: backend.server.v2.store.model.ReviewSubmissionRequest,
        user_id: str,
    ):
        return await backend.server.v2.admin.store_admin_routes.review_submission(
            request.store_listing_version_id, request, user_id
        )

    @staticmethod
    async def test_create_credentials(
        user_id: str,
        provider: ProviderName,
        credentials: Credentials,
    ) -> Credentials:
        from backend.server.integrations.router import (
            create_credentials,
            get_credential,
        )

        try:
            return await create_credentials(
                user_id=user_id, provider=provider, credentials=credentials
            )
        except Exception as e:
            logger.error(f"Error creating credentials: {e}")
            return await get_credential(
                provider=provider,
                user_id=user_id,
                cred_id=credentials.id,
            )

    def set_test_dependency_overrides(self, overrides: dict):
        app.dependency_overrides.update(overrides)
