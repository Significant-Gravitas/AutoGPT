from __future__ import annotations

import importlib
import inspect
import logging
from logging import Logger
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .base import BaseAgent


class AgentRegistry:
    agents: list[BaseAgent] = []
    # settings : AgentRegistrySettings

    def __init__(
        self,
    ):
        pass

    def add_agent(self, agent: BaseAgent):
        self.agents.append(agent)
