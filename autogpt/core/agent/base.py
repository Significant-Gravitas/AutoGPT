import abc
import logging
from pathlib import Path


class Agent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        ...

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        ...
