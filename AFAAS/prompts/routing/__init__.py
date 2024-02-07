from __future__ import annotations

from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    PromptStrategiesConfiguration,
)
from AFAAS.prompts.common.autocorrection import (
    AutoCorrectionStrategy,
    AutoCorrectionStrategyConfiguration,
    AutoCorrectionStrategyFunctionNames,
)
from AFAAS.prompts.routing.evaluate_and_select import (
    EvaluateSelectStrategy,
    EvaluateSelectStrategyConfiguration,
    EvaluateSelectStrategyFunctionNames,
)
from AFAAS.prompts.routing.routing import (
    RoutingStrategy,
    RoutingStrategyConfiguration,
    RoutingStrategyFunctionNames,
)
from AFAAS.prompts.routing.select_planning import (
    SelectPlanningStrategy,
    SelectPlanningStrategyConfiguration,
    SelectPlanningStrategyFunctionNames,
)


class StrategiesSetConfiguration(PromptStrategiesConfiguration):
    routing: RoutingStrategyConfiguration
    evaluate_and_select_approach: EvaluateSelectStrategyConfiguration
    select_planing_logic: SelectPlanningStrategyConfiguration
    autorection: AutoCorrectionStrategyConfiguration


class StrategiesSet:
    @staticmethod
    def get_strategies() -> list[AbstractPromptStrategy]:
        return [
            RoutingStrategy(**RoutingStrategy.default_configuration.dict()),
            EvaluateSelectStrategy(
                **EvaluateSelectStrategy.default_configuration.dict()
            ),
            AutoCorrectionStrategy(
                **AutoCorrectionStrategy.default_configuration.dict()
            ),
            SelectPlanningStrategy(
                **SelectPlanningStrategy.default_configuration.dict()
            ),
        ]
