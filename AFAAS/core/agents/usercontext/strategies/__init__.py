

from AFAAS.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextFunctionNames, RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration)
from AFAAS.core.prompting.base import (
    BasePromptStrategy, PromptStrategiesConfiguration)


class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from AFAAS.core.prompting.base import (
        AbstractPromptStrategy, BasePromptStrategy)

    @staticmethod
    def get_strategies(logger : Logger = Logger(__name__))-> list[BasePromptStrategy]:
        return [
            RefineUserContextStrategy(**RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
