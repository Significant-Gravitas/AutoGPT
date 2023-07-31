from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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

    def __str__(self) -> str:
        return (
            f"{self.description} (source: {self.source})\n"
            "```\n"
            f"{self.content}\n"
            "```"
        )


@dataclass
class FileContextItem(ContextItem):
    file_path: Path
    description: str

    @property
    def source(self) -> str:
        return f"local file '{self.file_path}'"

    @property
    def content(self) -> str:
        return self.file_path.read_text()


@dataclass
class FolderContextItem(ContextItem):
    path: Path

    def __post_init__(self) -> None:
        assert self.path.exists(), "Selected path does not exist"
        assert self.path.is_dir(), "Selected path is not a directory"

    @property
    def description(self) -> str:
        return f"The contents of the folder '{self.path}' in the workspace"

    @property
    def source(self) -> str:
        return f"local folder '{self.path}'"

    @property
    def content(self) -> str:
        items = [f"{p.name}{'/' if p.is_dir() else ''}" for p in self.path.iterdir()]
        items.sort()
        return "\n".join(items)


@dataclass
class StaticContextItem(ContextItem):
    description: str
    source: Optional[str]
    content: str
