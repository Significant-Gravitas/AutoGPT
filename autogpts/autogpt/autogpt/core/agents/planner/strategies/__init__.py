from logging import Logger

import autogpts.AFAAS.app.sdk.forge_log as agptlogger
from autogpts.autogpt.autogpt.core.agents.planner.strategies.initial_plan import (
    InitialPlanStrategy, InitialPlanStrategyConfiguration)
from autogpts.autogpt.autogpt.core.agents.planner.strategies.select_tool import (
    SelectToolStrategy, SelectToolStrategyConfiguration)
from autogpts.autogpt.autogpt.core.prompting.base import \
    PromptStrategiesConfiguration

LOG = agptlogger.ForgeLogger(__name__)


class StrategiesConfiguration(PromptStrategiesConfiguration):
    initial_plan: InitialPlanStrategyConfiguration
    select_tool: SelectToolStrategyConfiguration


class StrategiesSet:
    from autogpts.autogpt.autogpt.core.prompting.base import (
        AbstractPromptStrategy, BasePromptStrategy)

    @staticmethod
    def get_strategies(
        logger: Logger = Logger(__name__),
    ) -> list[AbstractPromptStrategy]:
        return [
            InitialPlanStrategy(
                logger=logger, **InitialPlanStrategy.default_configuration.dict()
            ),
            # SelectToolStrategy(
            #     logger=logger, **SelectToolStrategy.default_configuration.dict()
            # ),
        ]
