# sourcery skip: do-not-use-staticmethod
"""
Maintain backward Compatibility
"""
#from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Optional, Type

import distro
import yaml

from autogpt.prompts.generator import PromptGenerator
from autogpt.projects.agent_model import AgentModel


# Soon this will go in a folder where it remembers more stuff about the run(s)
SAVE_FILE = str(Path(os.getcwd()) / "ai_settings.yaml")


class AIConfig(AgentModel):  

    pass