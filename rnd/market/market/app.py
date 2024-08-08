import contextlib
import logging.config
import os

import dotenv
import fastapi
import fastapi.middleware.cors
import fastapi.middleware.gzip
import prisma
import prometheus_fastapi_instrumentator
import sentry_sdk
import sentry_sdk.integrations.asyncio
import sentry_sdk.integrations.fastapi
import sentry_sdk.integrations.starlette

import market.config
import market.routes.admin
import market.routes.agents
import market.routes.search

dotenv.load_dotenv()

logging.config.dictConfig(market.config.LogConfig().model_dump())

if os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        enable_tracing=True,
        environment=os.environ.get("RUN_ENV", default="CLOUD").lower(),
        integrations=[
            sentry_sdk.integrations.starlette.StarletteIntegration(
                transaction_style="url"
            ),
            sentry_sdk.integrations.fastapi.FastApiIntegration(transaction_style="url"),
            sentry_sdk.integrations.asyncio.AsyncioIntegration(),
        ],
    )

db_client = prisma.Prisma(auto_register=True)


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    await db_client.connect()
    yield
    await db_client.disconnect()


app = fastapi.FastAPI(
    title="Marketplace API",
    description="AutoGPT Marketplace API is a service that allows users to share AI agents.",
    summary="Maketplace API",
    version="0.1",
    lifespan=lifespan,
    root_path="/api/v1/market",
)

app.add_middleware(fastapi.middleware.gzip.GZipMiddleware, minimum_size=1000)
app.add_middleware(
    middleware_class=fastapi.middleware.cors.CORSMiddleware,
    allow_origins=[
        # Currently, we allow only next.js dev server
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(market.routes.agents.router, tags=["agents"])
app.include_router(market.routes.search.router, tags=["search"])
app.include_router(market.routes.admin.router, prefix="/admin", tags=["admin"])


@app.get("/health")
def health():
    return fastapi.responses.HTMLResponse(
        content="<h1>Marketplace API</h1>", status_code=200
    )


prometheus_fastapi_instrumentator.Instrumentator().instrument(app).expose(app)
