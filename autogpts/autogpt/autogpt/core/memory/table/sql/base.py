from __future__ import annotations

import abc
import uuid
import datetime
from pathlib import Path
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from pydantic import BaseModel
from ..base import AbstractTable



class BaseSQLTable(AbstractTable):
    def __init__(self) -> None:
        raise NotImplementedError()

    def add(self, value: dict) -> uuid.UUID:
        id = uuid.uuid4()
        value["id"] = id
        self.memory.add(key=id, value=value, table_name=self.table_name)
        return id

