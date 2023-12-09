from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from ..base import BaseLoop

if TYPE_CHECKING:
    from ..base import BaseAgent
    from AFAAS.core.resource.model_providers import ChatModelResponse

from .strategies import RefineUserContextFunctionNames


class UserContextLoop(BaseLoop):
    """A loop responsible for managing the user context in an agent.

    Args:
        agent (BaseAgent): The agent to which this loop is attached.

    Attributes:
        _active (bool): Indicates whether the loop is active or paused.
        loop_count (int): The number of loop iterations performed.
    
    Example:
        ```
        agent = MyCustomAgent()
        user_loop = UserContextLoop(agent)
        user_loop.run(
            agent,
            hooks,
            user_input_handler,
            user_message_handler
        )
        ```
    """


    class LoophooksDict(BaseLoop.LoophooksDict):
        pass

    def __init__(self) -> None:
        super().__init__()
        self._active = False

    async def run(
        self,
        agent: BaseAgent,
        hooks: LoophooksDict,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> dict:
        """Run the UserContextLoop to manage user context.

        Args:
            agent (BaseAgent): The agent instance.
            hooks (LoophooksDict): Dictionary containing hooks.
            user_input_handler (Callable[[str], Awaitable[str]]): A function to handle user input.
            user_message_handler (Callable[[str], Awaitable[str]]): A function to handle user messages.

        Returns:
            dict: A dictionary containing updated agent goals and sentences.

        Example:
            ```
            async def user_input_handler(input_str):
                # Handle user input and return a response.
                return "Response to user input"

            async def user_message_handler(message):
                # Handle user messages and return a response.
                return "Response to user message"

            agent = MyCustomAgent()
            user_loop = UserContextLoop(agent)
            result = await user_loop.run(
                agent,
                hooks,
                user_input_handler,
                user_message_handler
            )
            ```
        """
        self._agent._logger.info(f"Running UserContextLoop")

        self.loop_count = 0
        user_input = ""
        # _is_running is important because it avoid having two concurent loop in the same agent (cf : Agent.run())

        user_objectives: str = self._agent.agent_goal_sentence
        interupt_refinement_process: bool = False
        reformulated_goal: str = None
        while self._is_running:
            # if _active is false, then the loop is paused
            if self._active:
                self.loop_count += 1
                self._agent._logger.info(
                    f"Starting loop iteration number {self.loop_count}"
                )

                model_response : ChatModelResponse = await self._execute_strategy(
                    strategy_name="refine_user_context",
                    interupt_refinement_process=interupt_refinement_process,
                    user_objective=user_objectives,
                )

                if (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.REFINE_REQUIREMENTS
                ):
                    input_dict = model_response.parsed_result
                    reformulated_goal = model_response.parsed_result[
                        "reformulated_goal"
                    ]
                    user_objectives = await user_input_handler(input_dict)
                elif (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION
                ):
                    input_dict = model_response.parsed_result
                    user_objectives = await user_input_handler(input_dict)
                elif (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS
                    and reformulated_goal is not None
                ):
                    self._agent._logger.info(f"Exiting UserContextLoop")
                    return_value: dict = {
                        "agent_goal_sentence": reformulated_goal,
                        "agent_goals": model_response.parsed_result["goal_list"],
                    }
                    return return_value
                else:
                    self._agent._logger.error(model_response.parsed_result)
                    raise Exception

                if user_objectives.lower() == "y" and self.loop_count > 1:
                    interupt_refinement_process = True

                await self.save_agent() # TODO : self.save_agent()

    def __repr__(self):
        """Return a string representation of the UserContextLoop.

        Returns:
            str: A string representation of the UserContextLoop.
        """
        return "UserContextLoop()"
