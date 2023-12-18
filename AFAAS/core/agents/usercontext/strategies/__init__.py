from AFAAS.core.agents.usercontext.strategies.refine_user_context import (
    RefineUserContextFunctionNames,
    RefineUserContextStrategy,
    RefineUserContextStrategyConfiguration,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    PromptStrategiesConfiguration,
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
