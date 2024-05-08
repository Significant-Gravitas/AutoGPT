from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import orjson
from forge.config.config import Config

from ..memory_item import MemoryItem
from .base import VectorMemoryProvider

logger = logging.getLogger(__name__)


class JSONFileMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a JSON file"""

    SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, config: Config) -> None:
        """Initialize a class instance

        Args:
            config: Config object

        Returns:
            None
        """
        self.file_path = config.workspace_path / f"{config.memory_index}.json"
        self.file_path.touch()
        logger.debug(
            f"Initialized {__class__.__name__} with index path {self.file_path}"
        )

        self.memories = []
        try:
            self.load_index()
            logger.debug(f"Loaded {len(self.memories)} MemoryItems from file")
        except Exception as e:
            logger.warning(f"Could not load MemoryItems from file: {e}")
            self.save_index()

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter(self.memories)

    def __contains__(self, x: MemoryItem) -> bool:
        return x in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def add(self, item: MemoryItem):
        self.memories.append(item)
        logger.debug(f"Adding item to memory: {item.dump()}")
        self.save_index()
        return len(self.memories)

    def discard(self, item: MemoryItem):
        try:
            self.remove(item)
        except ValueError:  # item not in memory
            pass

    def clear(self):
        """Clears the data in memory."""
        self.memories.clear()
        self.save_index()

    def load_index(self):
        """Loads all memories from the index file"""
        if not self.file_path.is_file():
            logger.debug(f"Index file '{self.file_path}' does not exist")
            return
        with self.file_path.open("r") as f:
            logger.debug(f"Loading memories from index file '{self.file_path}'")
            json_index = orjson.loads(f.read())
            for memory_item_dict in json_index:
                self.memories.append(MemoryItem.parse_obj(memory_item_dict))

    def save_index(self):
        logger.debug(f"Saving memory index to file {self.file_path}")
        with self.file_path.open("wb") as f:
            return f.write(
                orjson.dumps(
                    [m.dict() for m in self.memories], option=self.SAVE_OPTIONS
                )
            )
