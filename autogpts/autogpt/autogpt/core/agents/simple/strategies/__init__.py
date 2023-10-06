from logging import Logger
from autogpt.core.prompting.base import PromptStrategiesConfiguration

from autogpt.core.agents.simple.strategies.initial_plan import (
    InitialPlanStrategy,
    InitialPlanStrategyConfiguration,
)
from autogpt.core.agents.simple.strategies.name_and_goals import (
    NameAndGoalsStrategy,
    NameAndGoalsConfiguration,
)
from autogpt.core.agents.simple.strategies.next_ability import (
    NextToolStrategy,
    NextToolConfiguration,
)
from autogpt.core.agents.simple.strategies.think import (
    ThinkStrategy,
    ThinkStrategyConfiguration,
)


class StrategiesConfiguration(PromptStrategiesConfiguration):
    name_and_goals: NameAndGoalsConfiguration
    initial_plan: InitialPlanStrategyConfiguration
    next_ability: NextToolConfiguration
    think: ThinkStrategyConfiguration


class Strategies:
    from autogpt.core.prompting.base import BasePromptStrategy, AbstractPromptStrategy

    @staticmethod
    def get_strategies(logger: Logger) -> list[AbstractPromptStrategy]:
        return [
            InitialPlanStrategy(
                logger=logger, **InitialPlanStrategy.default_configuration.dict()
            ),
            NameAndGoalsStrategy(
                logger=logger, **NameAndGoalsStrategy.default_configuration.dict()
            ),
            NextToolStrategy(
                logger=logger, **NextToolStrategy.default_configuration.dict()
            ),
            ThinkStrategy(logger=logger, **ThinkStrategy.default_configuration.dict()),
        ]
