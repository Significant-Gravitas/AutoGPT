from uuid import uuid4

from prisma import Prisma
from pydantic import BaseModel

prisma = Prisma(auto_register=True)


def connect():
    prisma.connect()


def disconnect():
    prisma.disconnect()


class BaseDbModel(BaseModel):
    id: str = ""

    def __init__(self, id: str = "", **data):
        data["id"] = id or str(uuid4())
        super().__init__(**data)
