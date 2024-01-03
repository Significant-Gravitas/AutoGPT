import json
from typing import Any, ClassVar, Generator, Optional

from AFAAS.configs.schema import AFAASModel
from AFAAS.interfaces.db.db import AbstractMemory
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class AFAASMessage(AFAASModel):
    message_id: str
    _table_name: ClassVar[str] = "message"
    task_id: Optional[str]


class AFAASMessageStack(AFAASModel):
    _messages: list[AFAASMessage] = []
    _memory: AbstractMemory = AbstractMemory.get_adapter()

    def add(self, message: AFAASMessage):
        self._messages.append(message)
        self._memory.get_table(message._table_name).add(message)
        return message.message_id

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not "_messages" in data.keys():
            self._messages = []

    def dict(self, *args, **kwargs) -> list[str]:
        return self._messages

    def json(self, *args, **kwargs):
        return json.dumps(self.dict())

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        LOG.error("Iterating over AFAASMessageStack")
        return iter(self._messages)

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
