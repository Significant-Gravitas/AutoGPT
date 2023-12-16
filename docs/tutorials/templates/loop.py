# Import necessary libraries and modules from the afaas framework and other packages.
from typing import Awaitable, Callable

from AFAAS.interfaces.agent import BaseAgent, BaseLoop
from AFAAS.interfaces.prompts.strategy import (LoophooksDict,
                                         PromptStrategiesConfiguration)


# Define your custom loop class. The class name should reflect its purpose.
class MyCustomLoop(BaseLoop):
    """A custom loop class for handling agent execution logic.
    Inherits from BaseLoop to access essential methods and attributes for agent execution.
    """

    def __init__(self, agent: BaseAgent) -> None:
        """Initialize the loop with an associated agent instance.

        Args:
            agent (BaseAgent): The agent instance associated with this loop.
        """
        super().__init__(agent)  # Call the parent class initializer
        self._active = False  # Set the loop to inactive by default

    async def run(
        self,
        agent: BaseAgent,
        hooks: LoophooksDict,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> dict:
        """Core method where the execution logic of the agent is defined.

        Args:
            agent (BaseAgent): The agent instance running the loop.
            hooks (LoophooksDict): A dictionary of hooks for user-defined actions/modifications.
            user_input_handler (Callable): An async function for handling user input.
            user_message_handler (Callable): An async function for handling user messages.

        Returns:
            dict: A dictionary containing any relevant information, e.g., final response, state, etc.
        """
        # This flag keeps the loop iterating. It could be used to keep the loop active until a certain condition is met.
        self._is_running = True

        while self._is_running:
            # Your code logic goes here.
            # This could include handling user input, interacting with the LLM, processing the model's response, etc.

            # Example: Getting user input
            user_input = await user_input_handler("Your question/query here")

            # Example: Handling the user input
            user_message = await user_message_handler(user_input)

            # Example: Executing a strategy to interact with the LLM
            model_response = await self.execute_strategy(
                strategy_name="your_strategy_name",
                user_message=user_message,
            )

            # Example: Parsing the model's response and determining the next action
            if model_response.parsed_result["condition"] == "some_condition":
                # Your logic based on the model's response
                pass

        # Optionally, save the agent's state before exiting the loop
        await self.save_agent()

        # Return any relevant information
        return {"final_response": "Your final response or information here"}

    async def save_agent(self):
        """An example method to save the current state of the agent.
        This could be implemented to save the agent's state to a file or a database.
        """
        # Your logic to save the agent's state goes here
        pass
