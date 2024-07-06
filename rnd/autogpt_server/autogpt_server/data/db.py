from uuid import uuid4
from prisma import Prisma
from pydantic import BaseModel

prisma = Prisma(auto_register=True)


async def connect():
    if not prisma.is_connected():
        await prisma.connect()


async def disconnect():
    if prisma.is_connected():
        await prisma.disconnect()


class BaseDbModel(BaseModel):
    id: str = ""

    def __init__(self, id: str = "", **data):
        data["id"] = id or str(uuid4())
        super().__init__(**data)
