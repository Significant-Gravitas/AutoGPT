from __future__ import annotations

import abc

from autogpt.core.messaging import MessageBroker
from autogpt.core.workspace import Workspace


class Agent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls, workpace_path: Workspace, message_broker: MessageBroker
    ) -> Agent:
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
