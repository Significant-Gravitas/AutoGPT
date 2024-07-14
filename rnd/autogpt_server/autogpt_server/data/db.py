from uuid import uuid4
from prisma import Prisma
from pydantic import BaseModel, field_validator

prisma = Prisma(auto_register=True)


async def connect():
    if not prisma.is_connected():
        await prisma.connect()


async def disconnect():
    if prisma.is_connected():
        await prisma.disconnect()


class BaseDbModel(BaseModel):
    id: str = ""

    @field_validator("id", mode="before")
    def set_graph_id(cls, id: str) -> str:
        return id or str(uuid4())
