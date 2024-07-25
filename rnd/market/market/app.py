import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from prisma import Prisma

from market.routes import agents

load_dotenv()
logger = logging.getLogger(__name__)

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
