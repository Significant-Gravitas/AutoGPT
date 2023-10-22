from logging import Logger

from autogpts.autogpt.autogpt.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextFunctionNames, RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration)
from autogpts.autogpt.autogpt.core.prompting.base import (
    BasePromptStrategy, PromptStrategiesConfiguration)


class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from autogpts.autogpt.autogpt.core.prompting.base import (
        AbstractPromptStrategy, BasePromptStrategy)

    @staticmethod
    def get_strategies(logger : Logger = Logger(__name__))-> list[BasePromptStrategy]:
        return [
            RefineUserContextStrategy(
                logger=logger, **RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
