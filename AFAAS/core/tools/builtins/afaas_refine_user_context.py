"""Tools to control the internal state of the program"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.agents.usercontext.main import UserContextAgent
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.tools.base import AbstractTool
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.interfaces.adapters import AbstractChatModelResponse
from AFAAS.prompts.usercontext.refine_user_context import (
    RefineUserContextFunctionNames,
)
from AFAAS.lib.utils.json_schema import JSONSchema
LOG = AFAASLogger(name=__name__)


@tool(
    name="afaas_refine_user_context",
    description="Assist user refining it's requirements thus improving LLM responses",
    parameters = {
        "user_objective": JSONSchema(
            type="string",
            description="The user's objective to be refined",
            examples=["I want to learn how to cook", "I want to learn how to cook"],
        )
    },
    hide=True,
    categories=["famework"],
)
async def afaas_refine_user_context(task: Task, agent: BaseAgent , user_objectives : str) -> None:
    """
    Configures the user context agent based on the current agent settings and executes the user context agent.
    Returns the updated agent goals.
    """
    try:
        from AFAAS.core.tools.builtins.user_interaction import user_interaction
        #user_objectives: str = agent.agent_goal_sentence
        interupt_refinement_process: bool = False
        reformulated_goal: str = None
        loop_count: int = 0
        while True:
                loop_count += 1
                LOG.info(f"Starting loop iteration number {loop_count}")

                model_response: AbstractChatModelResponse = (
                    await agent.execute_strategy(
                        strategy_name="refine_user_context",
                        interupt_refinement_process=interupt_refinement_process,
                        user_objective=user_objectives,
                        task=task, 
                    )
                )

                if (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.REFINE_REQUIREMENTS
                ):
                    input_dict = model_response.parsed_result
                    reformulated_goal = model_response.parsed_result[
                        "reformulated_goal"
                    ]
                    user_objectives = await  user_interaction(
                        query="\n".join(input_dict['questions']), 
                        task=task, 
                        agent=agent, 
                        skip_proxy=True
                        )
                elif (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.REQUEST_SECOND_CONFIRMATION
                ):
                    input_dict = model_response.parsed_result
                    user_objectives = await user_interaction(
                        query=input_dict, 
                        task=task, 
                        agent=agent, 
                        skip_proxy=True
                        )
                elif (
                    model_response.parsed_result["name"]
                    == RefineUserContextFunctionNames.VALIDATE_REQUIREMENTS
                    and reformulated_goal is not None
                ):
                    LOG.info(f"Exiting UserContextLoop")
                    task.memory["agent_goal_sentence"] =  reformulated_goal,
                    task.memory["agent_goals"] = model_response.parsed_result["goal_list"],
                    break
                else:
                    LOG.error(model_response.parsed_result)
                    raise Exception

                if user_objectives.lower() == "y" and loop_count > 1:
                    interupt_refinement_process = True

        # # FIXME:0.0.3: Move it outside of the tool
        # # Option 1 : Callback functions (one default to the tool & one custom for the plan)        
        # agent.agent_goal_sentence = return_value["agent_goal_sentence"]
        # agent.agent_goals = return_value["agent_goals"]
    except Exception as e:
        raise str(e)
