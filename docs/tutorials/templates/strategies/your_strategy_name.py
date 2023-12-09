# Import necessary libraries and modules from the afaas framework and other packages.
from AFAAS.core.core.resource.model_providers import ChatModelResponse
from AFAAS.core.core.resource.model_providers.chat_schema import ChatPrompt
from AFAAS.core.prompting.base import (BasePromptStrategy,
                                         PromptStrategiesConfiguration)
from pydantic import BaseModel


# Define a configuration class for your strategy. This will hold any custom configuration your strategy needs.
class YourStrategyNameConfiguration(PromptStrategiesConfiguration):
    """A Pydantic model to hold the configuration for YourStrategyName."""

    # Add any additional configuration properties here.
    # Example:
    # custom_config_property: str


# Define your custom strategy class. The class name should reflect its purpose.
class YourStrategyName(BasePromptStrategy):
    """A custom strategy class for handling specific interaction logic with the LLM.
    Inherits from BasePromptStrategy to access essential methods and attributes for building prompts.
    """

    # Override the default_configuration class method to provide your custom configuration.
    @classmethod
    def default_configuration(cls) -> YourStrategyNameConfiguration:
        """Provides the default configuration for this strategy.

        Returns:
            YourStrategyNameConfiguration: The default configuration object.
        """
        return YourStrategyNameConfiguration()

    def build_prompt(self, **kwargs) -> ChatPrompt:
        """Builds the chat prompt based on your strategy's logic.

        Args:
            **kwargs: Arbitrary keyword arguments. You can define any custom arguments your strategy needs.

        Returns:
            ChatPrompt: The chat prompt object to be sent to the LLM.
        """
        # Your code to build the chat prompt goes here.
        # This could include crafting messages, defining functions, etc.

        # Example: Creating messages and functions
        messages = []  # Replace with your list of ChatMessage objects
        functions = []  # Replace with your list of CompletionModelFunction objects

        # Create and return the ChatPrompt object
        return ChatPrompt(
            messages=messages,
            functions=functions,
        )

    def parse_response(self, response: ChatModelResponse) -> dict:
        """Parses the response from the LLM to extract meaningful information.

        Args:
            response (ChatModelResponse): The response object from the LLM.

        Returns:
            dict: A dictionary containing parsed information.
        """
        # Your code to parse the response goes here.
        # This could include extracting specific data from the response,
        # interpreting the model's answers, etc.

        # Example: Extracting a particular piece of information from the response
        parsed_info = {"extracted_data": response.parsed_result.get("some_key", None)}

        return parsed_info
