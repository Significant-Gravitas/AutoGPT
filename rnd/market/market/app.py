import contextlib
import os

import dotenv
import fastapi
import fastapi.middleware.gzip
import prisma
import sentry_sdk
import sentry_sdk.integrations.asyncio
import sentry_sdk.integrations.fastapi
import sentry_sdk.integrations.starlette

import market.routes.admin
import market.routes.agents
import market.routes.search

dotenv.load_dotenv()

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
)

app.add_middleware(fastapi.middleware.gzip.GZipMiddleware, minimum_size=1000)
app.include_router(
    market.routes.agents.router, prefix="/market/agents", tags=["agents"]
)
app.include_router(
    market.routes.search.router, prefix="/market/search", tags=["search"]
)
app.include_router(market.routes.admin.router, prefix="/market/admin", tags=["admin"])
