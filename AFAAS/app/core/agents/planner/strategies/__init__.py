from logging import Logger

from  AFAAS.app.sdk.forge_log import ForgeLogger
from AFAAS.app.core.agents.planner.strategies.initial_plan import (
    InitialPlanStrategy, InitialPlanStrategyConfiguration)
from AFAAS.app.core.agents.planner.strategies.select_tool import (
    SelectToolStrategy, SelectToolStrategyConfiguration)
from AFAAS.app.core.prompting.base import \
    PromptStrategiesConfiguration

LOG = ForgeLogger(__name__)


class StrategiesConfiguration(PromptStrategiesConfiguration):
    initial_plan: InitialPlanStrategyConfiguration
    select_tool: SelectToolStrategyConfiguration


class StrategiesSet:
    from AFAAS.app.core.prompting.base import (
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
