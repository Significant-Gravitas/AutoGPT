# Import necessary libraries, modules and classes.
from logging import Logger

from AFAAS.core.prompting.base import (BasePromptStrategy,
                                         PromptStrategiesConfiguration,
                                         PromptStrategy)

# Import your custom strategy class from the respective file.
from .your_strategy_name import YourStrategyName, YourStrategyNameConfiguration


# Define a configuration class for all the strategies. Each strategy gets its own configuration.
class StrategiesConfiguration(PromptStrategiesConfiguration):
    """A Pydantic model to hold configurations for all strategies."""

    # Define a configuration property for each strategy.
    your_strategy_name: YourStrategyNameConfiguration
    # Add other strategy configurations as properties here.


# Define a Strategies class to encapsulate all your strategies and provide them to the agent.
class Strategies:
    """Encapsulates all prompt strategies and provides a method to get them."""

    from AFAAS.core.prompting.base import BasePromptStrategy, PromptStrategy

    @staticmethod
    def get_strategies(logger: Logger) -> list[PromptStrategy]:
        """Provides a list of instantiated strategy objects to the agent.

        Args:
            logger (Logger): A logger instance for logging events during the execution.

        Returns:
            list[PromptStrategy]: A list of instantiated strategy objects.
        """
        # Instantiate and return a list of your strategy objects.
        # Use the logger and any necessary configuration from StrategiesConfiguration.
        return [
            YourStrategyName(
                **YourStrategyName.default_configuration.dict()
            ),
            # Instantiate other strategy objects and add them to the list.
        ]


# Note:
# - Each strategy should be defined in its own Python file within the strategies directory.
# - The StrategiesConfiguration class should have a property for each strategy's configuration.
# - The Strategies class should import and instantiate each strategy within the get_strategies method.
# - The get_strategies method is crucial as it provides the instantiated strategy objects to the agent.
# - Make sure to provide the necessary logger and configuration to each strategy when instantiating them.
