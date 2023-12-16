from __future__ import annotations

import copy

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from AFAAS.core.agents.planner.main import PlannerAgent

    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.lib.sdk import AFAASLogger
from AFAAS.core.lib.task.plan import Plan
from AFAAS.interfaces.configuration import UserConfigurable
from AFAAS.interfaces.prompts.schema import \
     PromptStrategyLanguageModelClassification
from AFAAS.interfaces.prompts.utils import \
    to_numbered_list
from AFAAS.interfaces.adapters import (
    ChatMessage, CompletionModelFunction)

from AFAAS.interfaces.prompts.strategy import (RESPONSE_SCHEMA, BasePromptStrategy,
                   PromptStrategiesConfiguration)

LOG = AFAASLogger(name=__name__)

class PlanningPromptStrategiesConfiguration(PromptStrategiesConfiguration):
    # DEFAULT_PROMPT_SCRATCHPAD = PromptScratchpad(
    #     tools= {},
    #     constraints= PLAN_PROMPT_CONSTRAINTS,
    #     resources=PLAN_PROMPT_RESOURCES,
    #     best_practices=PLAN_PROMPT_PERFORMANCE_EVALUATIONS,
    # )
    DEFAULT_BODY_TEMPLATE: str = (
        "## Constraints\n"
        "You operate within the following constraints:\n"
        "{constraints}\n"
        "\n"
        "## Resources\n"
        "You can leverage access to the following resources:\n"
        "{resources}\n"
        "\n"
        "## Commands\n"
        "You have access to the following commands:\n"
        "{tools}\n"
        "\n"
        "## Best practices\n"
        "{best_practices}"
    )

    DEFAULT_CHOOSE_ACTION_INSTRUCTION: str = (
        "Determine exactly one command to use next based on the given goals "
        "and the progress you have made so far, "
        "and respond using the JSON schema specified previously:"
    )
    DEFAULT_RESPONSE_SCHEMA = copy.deepcopy(RESPONSE_SCHEMA)

    body_template: str = UserConfigurable(default=DEFAULT_BODY_TEMPLATE)
    response_schema: dict = UserConfigurable(
        default_factory=DEFAULT_RESPONSE_SCHEMA.to_dict
    )
    choose_action_instruction: str = UserConfigurable(
        default=DEFAULT_CHOOSE_ACTION_INSTRUCTION
    )
    use_functions_api: bool = UserConfigurable(default=False)

    #########
    # State #
    #########
    progress_summaries: dict[tuple[int, int], str] = {(0, 0): ""}


class PlanningPromptStrategy(BasePromptStrategy):
    def __init__(
        self,
        model_classification:  PromptStrategyLanguageModelClassification,
        temperature: float,  # if coding 0.05
        top_p: Optional[float],
        max_tokens: Optional[int],
        frequency_penalty: Optional[float],  # Avoid repeting oneselfif coding 0.3
        presence_penalty: Optional[float],  # Avoid certain subjects
        **kwargs,
    ):
        super().__init__()
        self._prepend_messages: list[ChatMessage] = []
        self._append_messages: list[ChatMessage] = []

        self._model_classification = model_classification
        self._config = self.default_configuration

    # def construct_base_prompt(
    #     self,
    #     agent: PlannerAgent,
    #     prepend_messages: list[ChatMessage] = [],
    #     append_messages: list[ChatMessage] = [],
    #     reserve_tokens: int = 0,
    #     **kwargs
    # ) -> list[ChatMessage]:
    #     """Constructs and returns a prompt with the following structure:
    #     1. System prompt
    #     2. `prepend_messages`
    #     3. `append_messages`

    #     Params:
    #         prepend_messages: Messages to insert between the system prompt and message history
    #         append_messages: Messages to insert after the message history
    #         reserve_tokens: Number of tokens to reserve for content that is added later
    #     """

    #     # NOTE : PLANCE HOLDER
    #     # if agent.event_history:
    #     #     self._prepend_messages.insert(
    #     #         0,
    #     #         ChatMessage.system(
    #     #             "## Progress\n\n" f"{agent.event_history.fmt_paragraph()}"
    #     #         ),
    #     #     )

    #     messages: list[ChatMessage] = [
    #         ChatMessage.system(self._construct_system_prompt(agent=agent, **kwargs))
    #     ]
    #     if self._prepend_messages:
    #         messages.extend(self._prepend_messages)
    #     if self._append_messages:
    #         messages.extend(self._append_messages)

    #     return messages

    # FIXME : Uncompleted migration from AutoGPT Agent
    def _construct_system_prompt(
        self,
        agent: PlannerAgent,
        agent_directives: list,
        include_os_info: bool,
        tools: list,
        **kwargs,
    ) -> str:
        """Constructs a system prompt containing the most important information for the AI.

        Params:
            agent: The agent for which the system prompt is being constructed.

        Returns:
            str: The constructed system prompt.
        """
        reminder: str = (
            "## Constraints\n"
            "You operate within the following constraints:\n"
            "{constraints}\n"
            "\n"
            "## Resources\n"
            "You can leverage access to the following resources:\n"
            "{resources}\n"
            "\n"
            "## Commands\n"
            "You have access to the following commands:\n"
            "{tools}\n"
            "\n"
            "## Best practices\n"
            "{best_practices}"
        )

        LOG.trace(
            f"""
              DEBUG PLAN : Plan :\n{agent.plan.debug_dump(depth=1)}\n\n
              """
        )
        LOG.trace(
            f"""
              DEBUG PLAN : Plan :\n{Plan.debug_info_parse_task(agent.plan)}\n\n
              """
        )

        # plan_part =   [f"Plan :\n{agent.plan.dump(depth=1)}\n\n"] if agent.plan is not None else []
        plan_part = [
            f"Plan :\n {self.plan().generate_pitch(task =  self._agent.current_task)}"
        ]

        system_prompt_parts = (
            self._generate_intro_prompt(agent, **kwargs)
            + (self._generate_os_info(**kwargs) if include_os_info else [])
            + plan_part
            + [
                self._config.body_template.format(
                    constraints=to_numbered_list(
                        agent_directives.constraints
                        #    + self._generate_budget_constraint(agent.api_budget)
                    ),
                    resources=to_numbered_list(agent_directives.resources),
                    tools=self._generate_tools_list(tools),
                    best_practices=to_numbered_list(agent_directives.best_practices),
                )
            ]
            + self._generate_goals_info(agent)
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, system_prompt_parts)).strip("\n")
        # for plugin in agent.config.plugins:
        #     if not plugin.can_handle_post_prompt():
        #         continue
        #     plugin.post_prompt(self)

        # Construct full prompt
        from autogpt.AFAAS.interfaces.agent.assistants.prompt_manager import \
            get_os_info

        full_prompt_parts: list[str] = (
            self._generate_intro_prompt(agent=agent)
            + [
                f"The OS you are running on is: {get_os_info()}"
            ]  # NOTE : Should now be KWARG
            + self._generate_body(
                agent=agent,
            )  # additional_constraints=self._generate_budget_info(),
            + self._generate_goals_info(agent=agent)
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, full_prompt_parts)).strip("\n")

    #
    # _generate_os_info
    #
    def _generate_os_info(self, **kwargs) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """
        os_info = kwargs["os_info"]
        # os_info = (
        #     platform.platform(terse=True)
        #     if os_name != "Linux"
        #     else distro.name(pretty=True)
        # )
        return [f"The OS you are running on is: {os_info}"]

    #
    # _generate_tools_list
    #
    def _generate_tools_list(self, tools: list[CompletionModelFunction]) -> str:
        """Lists the tools available to the agent.

        Params:
            agent: The agent for which the tools are being listed.

        Returns:
            str: A string containing a numbered list of tools.
        """
        try:
            var = [cmd.fmt_line() for cmd in tools]
            return to_numbered_list(var)
        except AttributeError:
            LOG.warn(f"Formatting tools failed. {tools}")
            raise

    def _generate_intro_prompt(
        self, agent: Union[PlannerAgent, BaseAgent], **kwargs
    ) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {agent.agent_name}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    # FIXME :)
    def _generate_budget_info(self, agent: Union[PlannerAgent, BaseAgent]) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        if self.ai_config.api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${agent.ai_config.api_budget:.3f}"
            ]
        return []

    def _generate_goals_info(self, agent: PlannerAgent) -> list[str]:
        """Generates the goals information part of the prompt.

        Returns:
            str: The goals information part of the prompt.
        """
        if agent.agent_goals:
            return [
                "\n".join(
                    [
                        "## Goals",
                        "For your task, you must fulfill the following goals:",
                        to_numbered_list(agent.agent_goals),
                    ]
                )
            ]
        return []

    # def _generate_body(
    #     self,
    #     agent: Union[PlannerAgent, BaseAgent],
    #     *,
    #     additional_constraints: list[str] = [],
    #     additional_resources: list[str] = [],
    #     additional_best_practices: list[str] = [],
    # ) -> list[str]:
    #     """
    #     Generates a prompt section containing the constraints, tools, resources,
    #     and best practices.

    #     Params:
    #         agent: The agent for which the prompt string is being generated.
    #         additional_constraints: Additional constraints to be included in the prompt string.
    #         additional_resources: Additional resources to be included in the prompt string.
    #         additional_best_practices: Additional best practices to be included in the prompt string.

    #     Returns:
    #         str: The generated prompt section.
    #     """
    #     body: list[str] = []

    #     # NOTE : if agent.constraints :
    #     #     body.append("## Constraints\n"
    #     #     "You operate within the following constraints:\n"
    #     #     f"{to_numbered_list(agent.constraints + additional_constraints)}")

    #     # NOTE : PLACE HOLDER
    #     # if agent.resources :
    #     #     body.append("## Resources\n"
    #     #     "You can leverage access to the following resources:\n"
    #     #     f"{to_numbered_list(agent.resources + additional_resources)}")

    #     body.append(
    #         "## Abilities\n"
    #         "You have access to the following commands:\n"
    #         f"{self._list_tools(agent)}"
    #     )

    #     # NOTE : PLANCE HOLDER
    #     # if agent.best_practices :
    #     #     body.append("## Best practices\n"
    #     #     f"{to_numbered_list(agent.best_practices + additional_best_practices)}")

    #     return body

    def _list_tools(self, agent: Union[PlannerAgent, BaseAgent]) -> str:
        """Lists the tools available to the agent.

        Params:
            agent: The agent for which the tools are being listed.

        Returns:
            str: A string containing a numbered list of tools.
        """
        if agent._tool_registry:
            return to_numbered_list(agent._tool_registry.list_tools_descriptions())

        return []

    # FIXME : Implement this to converge to AutoGPT vs Loose "default_tool_choice functionality" => Move to OpenAPIProvider safer
    # #############
    # # Utilities #
    # #############
    # @staticmethod
    # def extract_command(
    #     assistant_reply_json: dict,
    #     assistant_reply: AssistantChatMessageDict,
    #     use_openai_functions_api: bool,
    # ) -> tuple[str, dict[str, str]]:
    #     """Parse the response and return the command name and arguments

    #     Args:
    #         assistant_reply_json (dict): The response object from the AI
    #         assistant_reply (ChatModelResponse): The model response from the AI
    #         config (Config): The config object

    #     Returns:
    #         tuple: The command name and arguments

    #     Raises:
    #         json.decoder.JSONDecodeError: If the response is not valid JSON

    #         Exception: If any other error occurs
    #     """
    #     if use_openai_functions_api:
    #         if "tool_calls" not in assistant_reply:
    #             raise InvalidAgentResponseError("No 'function_call' in assistant reply")
    #         assistant_reply_json["command"] = {
    #             "name": assistant_reply["tool_calls"]["name"],
    #             "args": json.loads(assistant_reply["tool_calls"]["arguments"]),
    #         }
    #     try:
    #         if not isinstance(assistant_reply_json, dict):
    #             raise InvalidAgentResponseError(
    #                 f"The previous message sent was not a dictionary {assistant_reply_json}"
    #             )

    #         if "command" not in assistant_reply_json:
    #             raise InvalidAgentResponseError("Missing 'command' object in JSON")

    #         command = assistant_reply_json["command"]
    #         if not isinstance(command, dict):
    #             raise InvalidAgentResponseError("'command' object is not a dictionary")

    #         if "name" not in command:
    #             raise InvalidAgentResponseError("Missing 'name' field in 'command' object")

    #         command_name = command["name"]

    #         # Use an empty dictionary if 'args' field is not present in 'command' object
    #         arguments = command.get("args", {})

    #         return command_name, arguments

    #     except json.decoder.JSONDecodeError:
    #         raise InvalidAgentResponseError("Invalid JSON")

    #     except Exception as e:
    #         raise InvalidAgentResponseError(str(e))
