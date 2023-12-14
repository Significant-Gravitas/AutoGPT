from logging import Logger

from  AFAAS.core.lib.sdk.logger import AFAASLogger
from AFAAS.core.agents.planner.strategies.initial_plan import (
    InitialPlanStrategy, InitialPlanStrategyConfiguration)
from AFAAS.core.agents.planner.strategies.select_tool import (
    SelectToolStrategy, SelectToolStrategyConfiguration)
from AFAAS.core.prompting.base import \
    PromptStrategiesConfiguration

LOG = AFAASLogger(name=__name__)


class StrategiesConfiguration(PromptStrategiesConfiguration):
    initial_plan: InitialPlanStrategyConfiguration
    select_tool: SelectToolStrategyConfiguration


class StrategiesSet:
    from AFAAS.core.prompting.base import (
        AbstractPromptStrategy, BasePromptStrategy)

    @staticmethod
    def get_strategies(
        logger: Logger = Logger(__name__),
    ) -> list[BasePromptStrategy]:
        return [
            InitialPlanStrategy(
                logger=logger, **InitialPlanStrategy.default_configuration.dict()
            ),
            # SelectToolStrategy(
            #     logger=logger, **SelectToolStrategy.default_configuration.dict()
            # ),
        ]
