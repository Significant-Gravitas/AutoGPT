import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from forge.file_storage.base import FileStorage
from forge.utils.file_operations import decode_textual_file

logger = logging.getLogger(__name__)


class BaseContextItem(ABC):
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the context item"""
        ...

    @property
    @abstractmethod
    def source(self) -> Optional[str]:
        """A string indicating the source location of the context item"""
        ...

    @abstractmethod
    def get_content(self, workspace: FileStorage) -> str:
        """The content represented by the context item"""
        ...

    def fmt(self, workspace: FileStorage) -> str:
        return (
            f"{self.description} (source: {self.source})\n"
            "```\n"
            f"{self.get_content(workspace)}\n"
            "```"
        )


class FileContextItem(BaseModel, BaseContextItem):
    path: Path
    type: Literal["file"] = "file"

    @property
    def description(self) -> str:
        return f"The current content of the file '{self.path}'"

    @property
    def source(self) -> str:
        return str(self.path)

    def get_content(self, workspace: FileStorage) -> str:
        with workspace.open_file(self.path, "r", True) as file:
            return decode_textual_file(file, self.path.suffix, logger)


class FolderContextItem(BaseModel, BaseContextItem):
    path: Path
    type: Literal["folder"] = "folder"

    @property
    def description(self) -> str:
        return f"The contents of the folder '{self.path}' in the workspace"

    @property
    def source(self) -> str:
        return str(self.path)

    def get_content(self, workspace: FileStorage) -> str:
        files = [str(p) for p in workspace.list_files(self.path)]
        folders = [f"{str(p)}/" for p in workspace.list_folders(self.path)]
        items = folders + files
        items.sort()
        return "\n".join(items)


class StaticContextItem(BaseModel, BaseContextItem):
    item_description: str = Field(alias="description")
    item_source: Optional[str] = Field(alias="source")
    item_content: str = Field(alias="content")
    type: Literal["static"] = "static"


ContextItem = FileContextItem | FolderContextItem | StaticContextItem
