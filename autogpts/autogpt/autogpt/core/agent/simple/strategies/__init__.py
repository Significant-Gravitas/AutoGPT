from logging import Logger
from autogpt.core.prompting.base import PromptStrategiesConfiguration

from autogpt.core.agent.simple.strategies.initial_plan import (
    InitialPlan,
    InitialPlanConfiguration,
)
from autogpt.core.agent.simple.strategies.name_and_goals import (
    NameAndGoals,
    NameAndGoalsConfiguration,
)
from autogpt.core.agent.simple.strategies.next_ability import (
    NextAbility,
    NextAbilityConfiguration,
)
from autogpt.core.agent.simple.strategies.think import (ThinkStrategy, ThinkStrategyConfiguration)

class StrategiesConfiguration(PromptStrategiesConfiguration):
    name_and_goals: NameAndGoalsConfiguration
    initial_plan: InitialPlanConfiguration
    next_ability: NextAbilityConfiguration
    think : ThinkStrategyConfiguration



class Strategies:
    from autogpt.core.prompting.base import BasePromptStrategy, PromptStrategy

    @staticmethod
    def get_strategies(logger: Logger) -> list[PromptStrategy]:
        return [
            InitialPlan(logger=logger, **InitialPlan.default_configuration.dict()),
            NameAndGoals(logger=logger, **NameAndGoals.default_configuration.dict()),
            NextAbility(logger=logger, **NextAbility.default_configuration.dict()),
            ThinkStrategy(logger=logger, **ThinkStrategy.default_configuration.dict())
        ]
