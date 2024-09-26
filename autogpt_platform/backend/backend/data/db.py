import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from prisma import Prisma
from pydantic import BaseModel, Field, field_validator

from backend.util.retry import logged_retry

load_dotenv()

PRISMA_SCHEMA = os.getenv("PRISMA_SCHEMA", "schema.prisma")
os.environ["PRISMA_SCHEMA_PATH"] = PRISMA_SCHEMA

prisma, conn_id = Prisma(auto_register=True), ""

logger = logging.getLogger(__name__)


async def connect():
    if prisma.is_connected():
        return

    global conn_id
    if not conn_id:
        conn_id = str(uuid4())

    await logged_retry(
        func=prisma.connect,
        resource_name=f"Prisma-{conn_id}",
        action_name="Acquiring connection",
    )


async def disconnect():
    if not prisma.is_connected():
        return

    await logged_retry(
        func=prisma.disconnect,
        resource_name=f"Prisma-{conn_id}",
        action_name="Releasing connection",
    )


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
