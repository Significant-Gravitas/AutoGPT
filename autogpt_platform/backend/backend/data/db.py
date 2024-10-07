import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from prisma import Prisma
from pydantic import BaseModel, Field, field_validator

from backend.util.retry import conn_retry

load_dotenv()

PRISMA_SCHEMA = os.getenv("PRISMA_SCHEMA", "schema.prisma")
os.environ["PRISMA_SCHEMA_PATH"] = PRISMA_SCHEMA

prisma = Prisma(auto_register=True)

logger = logging.getLogger(__name__)


@conn_retry("Prisma", "Acquiring connection")
async def connect():
    if prisma.is_connected():
        return
    await prisma.connect()


@conn_retry("Prisma", "Releasing connection")
async def disconnect():
    if not prisma.is_connected():
        return
    await prisma.disconnect()


@asynccontextmanager
async def transaction():
    async with prisma.tx() as tx:
        yield tx


class BaseDbModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("id", mode="before")
    def set_model_id(cls, id: str) -> str:
        # In case an empty ID is submitted
        return id or str(uuid4())
