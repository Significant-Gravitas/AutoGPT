from __future__ import annotations

from AFAAS.interfaces.prompts.strategy import PromptStrategiesConfiguration
from AFAAS.prompts.usercontext.refine_user_context import (
    RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration,
)


class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

    @staticmethod
    def get_strategies() -> list[AbstractPromptStrategy]:
        return [
            RefineUserContextStrategy(
                **RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
