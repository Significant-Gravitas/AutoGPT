from uuid import uuid4

from prisma import Prisma
from pydantic import BaseModel, Field, field_validator
import os
from dotenv import load_dotenv

load_dotenv()

# Use the DATABASE_URL from env vars, defaults to SQLlite
DATABASE_URL = os.getenv('DATABASE_URL', 'file:./database.db')

prisma = Prisma(auto_register=True, datasource={"url": DATABASE_URL})


async def connect():
    if not prisma.is_connected():
        await prisma.connect()


async def disconnect():
    if prisma.is_connected():
        await prisma.disconnect()


class BaseDbModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("id", mode="before")
    def set_model_id(cls, id: str) -> str:
        # In case an empty ID is submitted
        return id or str(uuid4())
