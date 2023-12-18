from __future__ import annotations

from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy
import AFAAS.prompts.planner as planner


def load_all_strategies() -> list[AbstractPromptStrategy]:
    return planner.StrategiesSet.get_strategies()
