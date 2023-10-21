from logging import Logger
from autogpt.core.prompting.base import (
    BasePromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration,
    RefineUserContextFunctionNames,
)

class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from autogpt.core.prompting.base import BasePromptStrategy, AbstractPromptStrategy

    @staticmethod
    def get_strategies(logger=Logger) -> list[BasePromptStrategy]:
        return [
            RefineUserContextStrategy(
                logger=logger, **RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
