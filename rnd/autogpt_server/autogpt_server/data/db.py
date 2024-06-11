import asyncio
from uuid import uuid4
from prisma import Prisma
from pydantic import BaseModel

prisma = Prisma(auto_register=True)


def connect_sync():
    asyncio.get_event_loop().run_until_complete(connect())


async def connect():
    await prisma.connect()


async def disconnect():
    await prisma.disconnect()


class BaseDbModel(BaseModel):
    id: str = ""

    def __init__(self, id: str = "", **data):
        data["id"] = id or str(uuid4())
        super().__init__(**data)
