from __future__ import annotations

from pathlib import Path
from typing import Iterator

import orjson

from autogpt.config import Config
from autogpt.logs import logger

from ..memory_item import MemoryItem
from .base import VectorMemoryProvider


class JSONFileMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a JSON file"""

    SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, cfg: Config) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        workspace_path = Path(cfg.workspace_path)
        self.file_path = workspace_path / f"{cfg.memory_index}.json"
        self.file_path.touch()
        logger.debug(f"Initialized {__name__} with index path {self.file_path}")

        self.memories = []
        self.save_index()

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self.memories)

    def __contains__(self, x: MemoryItem) -> bool:
        return x in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def add(self, item: MemoryItem):
        self.memories.append(item)
        self.save_index()
        return len(self.memories)

    def discard(self, item: MemoryItem):
        try:
            self.remove(item)
        except:
            pass

    def clear(self):
        """Clears the data in memory."""
        self.memories.clear()
        self.save_index()

    def save_index(self):
        logger.debug(f"Saving memory index to file {self.file_path}")
        with self.file_path.open("wb") as f:
            return f.write(orjson.dumps(self.memories, option=self.SAVE_OPTIONS))
