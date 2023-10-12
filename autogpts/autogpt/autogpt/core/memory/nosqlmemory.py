from __future__ import annotations

from typing import TYPE_CHECKING

from autogpt.core.configuration import Configurable
from autogpt.core.memory.base import AbstractMemory

if TYPE_CHECKING:
    from autogpt.core.memory.table.base import BaseTable


class NoSQLMemory(AbstractMemory):
    def get_table(self, table_name: str) -> BaseTable:
        if self.__class__ == NoSQLMemory:
            raise TypeError(
                "get_table method cannot be called on NoSQLMemory class directly"
            )
        return super().get_table(table_name)
