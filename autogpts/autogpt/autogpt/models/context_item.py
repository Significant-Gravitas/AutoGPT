import logging
import os.path
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from autogpt.commands.file_operations_utils import decode_textual_file

logger = logging.getLogger(__name__)


class ContextItem(ABC):
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

    @property
    @abstractmethod
    def content(self) -> str:
        """The content represented by the context item"""
        ...

    def fmt(self) -> str:
        return (
            f"{self.description} (source: {self.source})\n"
            "```\n"
            f"{self.content}\n"
            "```"
        )


class FileContextItem(BaseModel, ContextItem):
    file_path_in_workspace: Path
    workspace_path: Path

    @property
    def file_path(self) -> Path:
        return self.workspace_path / self.file_path_in_workspace

    @property
    def description(self) -> str:
        return f"The current content of the file '{self.file_path_in_workspace}'"

    @property
    def source(self) -> str:
        return str(self.file_path_in_workspace)

    @property
    def content(self) -> str:
        # TODO: use workspace.open_file()
        with open(self.file_path, "rb") as file:
            return decode_textual_file(file, os.path.splitext(file.name)[1], logger)


class FolderContextItem(BaseModel, ContextItem):
    path_in_workspace: Path
    workspace_path: Path

    @property
    def path(self) -> Path:
        return self.workspace_path / self.path_in_workspace

    def __post_init__(self) -> None:
        assert self.path.exists(), "Selected path does not exist"
        assert self.path.is_dir(), "Selected path is not a directory"

    @property
    def description(self) -> str:
        return f"The contents of the folder '{self.path_in_workspace}' in the workspace"

    @property
    def source(self) -> str:
        return str(self.path_in_workspace)

    @property
    def content(self) -> str:
        items = [f"{p.name}{'/' if p.is_dir() else ''}" for p in self.path.iterdir()]
        items.sort()
        return "\n".join(items)


class StaticContextItem(BaseModel, ContextItem):
    item_description: str = Field(alias="description")
    item_source: Optional[str] = Field(alias="source")
    item_content: str = Field(alias="content")
