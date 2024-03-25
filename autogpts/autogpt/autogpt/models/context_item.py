import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from autogpt.agents.base import BaseAgent
from autogpt.agents.features.agent_file_manager import AgentFileManagerMixin
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

    @abstractmethod
    def get_content(self, agent: BaseAgent) -> str:
        """The content represented by the context item"""
        ...

    def fmt(self, agent: BaseAgent) -> str:
        return (
            f"{self.description} (source: {self.source})\n"
            "```\n"
            f"{self.get_content(agent)}\n"
            "```"
        )


class FileContextItem(BaseModel, ContextItem):
    path: Path

    @property
    def description(self) -> str:
        return f"The current content of the file '{self.path}'"

    @property
    def source(self) -> str:
        return str(self.path)

    def get_content(self, agent: BaseAgent) -> str:
        if not isinstance(agent, AgentFileManagerMixin):
            raise ValueError("Agent must implement AgentFileManagerMixin")

        with agent.workspace.open_file(self.path, "r", True) as file:
            return decode_textual_file(file, self.path.suffix, logger)


class FolderContextItem(BaseModel, ContextItem):
    path: Path

    @property
    def description(self) -> str:
        return f"The contents of the folder '{self.path}' in the workspace"

    @property
    def source(self) -> str:
        return str(self.path)

    def get_content(self, agent: BaseAgent) -> str:
        if not isinstance(agent, AgentFileManagerMixin):
            raise ValueError("Agent must implement AgentFileManagerMixin")

        files = [str(p) for p in agent.workspace.list_files(self.path)]
        folders = [f"{str(p)}/" for p in agent.workspace.list_folders(self.path)]
        items = folders + files
        items.sort()
        return "\n".join(items)


class StaticContextItem(BaseModel, ContextItem):
    item_description: str = Field(alias="description")
    item_source: Optional[str] = Field(alias="source")
    item_content: str = Field(alias="content")
