from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generator, Optional, Type

from AFAAS.configs.schema import AFAASModel
from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent


class AFAASMessage(ABC, AFAASModel):
    message_id: str
    _table_name: ClassVar[str]  # = "message"
    task_id: Optional[str]


class AFAASMessageStack(AFAASModel):
    _messages: dict[AFAASMessage] = {}
    db: AbstractMemory

    async def db_create(self, message: AFAASMessage):
        self._messages[message.message_id] = message
        table = await self.db.get_table(message._table_name)
        await table.add(value=message, id=message.message_id)
        return message.message_id

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not "_messages" in data.keys():
            self._messages = {}

    def dict(self, *args, **kwargs) -> dict[str]:
        return self._messages

    def json(self, *args, **kwargs):
        return json.dumps(self.dict())

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages.items())

    @classmethod
    def __get_validators__(cls) -> Generator:
        LOG.trace(f"{cls.__name__}.__get_validators__()")
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> "AFAASMessageStack":
        LOG.trace(f"{cls.__name__}.validate()")
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            return cls(**v)
        else:
            raise TypeError(f"Expected AFAASMessageStack or dict, received {type(v)}")

    def __str__(self):
        return self._messages.__str__()

    async def load(
        self, agent: BaseAgent, cls: Type[AFAASMessage]
    ) -> dict[AFAASMessage]:
        from AFAAS.interfaces.db.db_table import AbstractTable

        table = await agent.db.get_table(cls._table_name)
        list = await table.list(
            filter=AbstractTable.FilterDict(
                {
                    "agent_id": [
                        AbstractTable.FilterItem(
                            value=str(agent.agent_id),
                            operator=AbstractTable.Operators.EQUAL_TO,
                        )
                    ]
                }
            ),
            order_column="created_at",
            order_direction="desc",
        )
        for message in list:
            if message["message_id"] not in self._messages.keys():
                self._messages[message["message_id"]] = cls(**message)
            else:
                LOG.warning(f"Message {message['message_id']} already loaded")
        return self
