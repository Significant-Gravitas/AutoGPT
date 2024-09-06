import asyncio
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from prisma import Prisma
from pydantic import BaseModel, Field, field_validator

load_dotenv()

PRISMA_SCHEMA = os.getenv("PRISMA_SCHEMA", "schema.prisma")
os.environ["PRISMA_SCHEMA_PATH"] = PRISMA_SCHEMA

prisma, conn_id = Prisma(auto_register=True), ""

logger = logging.getLogger(__name__)


async def connect(call_count=0):
    global conn_id
    if not conn_id:
        conn_id = str(uuid4())

    try:
        logger.info(f"[Prisma-{conn_id}] Acquiring connection..")
        if not prisma.is_connected():
            await prisma.connect()
        logger.info(f"[Prisma-{conn_id}] Connection acquired!")
    except Exception as e:
        if call_count <= 5:
            logger.info(f"[Prisma-{conn_id}] Connection failed: {e}. Retrying now..")
            await asyncio.sleep(call_count)
            await connect(call_count + 1)
        else:
            raise e


async def disconnect():
    if prisma.is_connected():
        logger.info(f"[Prisma-{conn_id}] Releasing connection.")
        await prisma.disconnect()
        logger.info(f"[Prisma-{conn_id}] Connection released.")


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
