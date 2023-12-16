

from AFAAS.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextFunctionNames, RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration)
from AFAAS.interfaces.prompts.strategy import (
    BasePromptStrategy, PromptStrategiesConfiguration)


class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    refine_user_context: RefineUserContextStrategyConfiguration


class StrategiesSet:
    from AFAAS.interfaces.prompts.strategy import (
        AbstractPromptStrategy, BasePromptStrategy)

    @staticmethod
    def get_strategies()-> list[BasePromptStrategy]:
        return [
            RefineUserContextStrategy(**RefineUserContextStrategy.default_configuration.dict()
            ),
        ]
