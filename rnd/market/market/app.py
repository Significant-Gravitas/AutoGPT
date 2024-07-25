
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from prisma import Prisma
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from market.routes import agents

load_dotenv()

if os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"), 
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
        enable_tracing=True,
        environment=os.environ.get("RUN_ENV", default="CLOUD").lower(),
        integrations=[
            StarletteIntegration(transaction_style="url"),
            FastApiIntegration(transaction_style="url"),
            AsyncioIntegration(),
        ],
    )

db_client = Prisma(auto_register=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_client.connect()
    yield
    await db_client.disconnect()


app = FastAPI(
    title="Marketplace  API",
    description=(
        "AutoGPT Marketplace API is a service that allows users to share AI agents."
    ),
    summary="Maketplace API",
    version="0.1",
    lifespan=lifespan,
)

# Add gzip middleware to compress responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


app.include_router(agents.router, prefix="/market/agents", tags=["agents"])
