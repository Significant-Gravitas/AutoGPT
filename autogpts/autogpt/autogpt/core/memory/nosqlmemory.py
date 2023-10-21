from __future__ import annotations

from typing import TYPE_CHECKING

from autogpts.autogpt.autogpt.core.configuration import Configurable
from autogpts.autogpt.autogpt.core.memory.base import AbstractMemory

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.memory.table.base import AbstractTable


class NoSQLMemory(AbstractMemory):
    def get_table(self, table_name: str) -> AbstractTable:
        if self.__class__ == NoSQLMemory:
            raise TypeError(
                "get_table method cannot be called on NoSQLMemory class directly"
            )
        return super().get_table(table_name)
