from __future__ import annotations

from AFAAS.interfaces.prompts.strategy import PromptStrategiesConfiguration
from AFAAS.lib.sdk.logger import AFAASLogger

from AFAAS.prompts.planner.select_tool import SelectToolStrategyConfiguration

LOG = AFAASLogger(name=__name__)


class StrategiesConfiguration(PromptStrategiesConfiguration):
    select_tool: SelectToolStrategyConfiguration


class StrategiesSet:
    from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

    @staticmethod
    def get_strategies() -> list[AbstractPromptStrategy]:
        return [
            # SelectToolStrategy(
            #     **SelectToolStrategy.config.dict()
            # ),
        ]
