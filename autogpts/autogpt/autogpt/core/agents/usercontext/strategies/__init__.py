from logging import Logger
from autogpts.autogpt.autogpt.core.prompting.base import (
    BasePromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpts.autogpt.autogpt.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration,
    RefineUserContextFunctionNames,
)

class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from autogpts.autogpt.autogpt.core.prompting.base import BasePromptStrategy, AbstractPromptStrategy

    @staticmethod
    def get_strategies(logger : Logger = Logger(__name__))-> list[BasePromptStrategy]:
        return [
            RefineUserContextStrategy(
                logger=logger, **RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
