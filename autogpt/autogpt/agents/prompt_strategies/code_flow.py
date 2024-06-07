import re
from logging import Logger
from typing import Iterable, Sequence

from forge.command import Command
from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.json.parsing import extract_dict_from_json
from forge.llm.prompting import ChatPrompt, LanguageModelClassification, PromptStrategy
from forge.llm.prompting.utils import indent
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    ChatMessage,
)
from forge.models.config import SystemConfiguration
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import InvalidAgentResponseError
from forge.utils.function.code_validation import CodeValidator
from forge.utils.function.model import FunctionDef
from pydantic import BaseModel, Field

from autogpt.agents.prompt_strategies.one_shot import (
    AssistantThoughts,
    OneShotAgentActionProposal,
    OneShotAgentPromptConfiguration,
)

_RESPONSE_INTERFACE_NAME = "AssistantResponse"


class CodeFlowAgentActionProposal(BaseModel):
    thoughts: AssistantThoughts
    immediate_plan: str = Field(
        ...,
        description="We will be running an iterative process to execute the plan, "
        "Write the partial / immediate plan to execute your plan as detailed and "
        "efficiently as possible without the help of the reasoning/intelligence. "
        "The plan should describe the output of the immediate plan, so that the next "
        "iteration can be executed by taking the output into account. "
        "Try to do as much as possible without making any assumption or uninformed "
        "guesses. Avoid large output at all costs!!!\n"
        "Format: Objective[Objective of this iteration, explain what's the use of this "
        "iteration for the next one] Plan[Plan that does not require any reasoning or "
        "intelligence] Output[Output of the plan / should be small, avoid whole file "
        "output]",
    )
    python_code: str = Field(
        ...,
        description=(
            "Write the fully-functional Python code of the immediate plan. "
            "The output will be an `async def main() -> str` function of the immediate "
            "plan that return the string output, the output will be passed into the "
            "LLM context window so avoid returning the whole content!. "
            "Use ONLY the listed available functions and built-in Python features. "
            "Leverage the given magic functions to implement function calls for which "
            "the arguments can't be determined yet. "
            "Example:`async def main() -> str:\n"
            "    return await provided_function('arg1', 'arg2').split('\\n')[0]`"
        ),
    )


FINAL_INSTRUCTION: str = (
    "You have to give the answer in the from of JSON schema specified previously. "
    "For the `python_code` field, you have to write Python code to execute your plan "
    "as efficiently as possible. Your code will be executed directly without any "
    "editing, if it doesn't work you will be held responsible. "
    "Use ONLY the listed available functions and built-in Python features. "
    "Do not make uninformed assumptions "
    "(e.g. about the content or format of an unknown file). Leverage the given magic "
    "functions to implement function calls for which the arguments can't be determined "
    "yet. Reduce the amount of unnecessary data passed into these magic functions "
    "where possible, because magic costs money and magically processing large amounts "
    "of data is expensive. If you think are done with the task, you can simply call "
    "finish(reason='your reason') to end the task, "
    "a function that has one `finish` command, don't mix finish with other functions! "
    "If you still need to do other functions, "
    "let the next cycle execute the `finish` function. "
    "Avoid hard-coding input values as input, and avoid returning large outputs. "
    "The code that you have been executing in the past cycles can also be buggy, "
    "so if you see undesired output, you can always try to re-plan, and re-code. "
)


class CodeFlowAgentPromptStrategy(PromptStrategy):
    default_configuration: OneShotAgentPromptConfiguration = (
        OneShotAgentPromptConfiguration()
    )

    def __init__(
        self,
        configuration: SystemConfiguration,
        logger: Logger,
    ):
        self.config = configuration
        self.response_schema = JSONSchema.from_dict(
            CodeFlowAgentActionProposal.schema()
        )
        self.logger = logger
        self.commands: Sequence[Command] = []  # Sequence -> disallow list modification

    @property
    def model_classification(self) -> LanguageModelClassification:
        return LanguageModelClassification.FAST_MODEL  # FIXME: dynamic switching

    def build_prompt(
        self,
        *,
        messages: list[ChatMessage],
        task: str,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: Sequence[Command],
        **extras,
    ) -> ChatPrompt:
        """Constructs and returns a prompt with the following structure:
        1. System prompt
        3. `cycle_instruction`
        """
        system_prompt, response_prefill = self.build_system_prompt(
            ai_profile=ai_profile,
            ai_directives=ai_directives,
            commands=commands,
        )

        self.commands = commands
        final_instruction_msg = ChatMessage.system(FINAL_INSTRUCTION)

        return ChatPrompt(
            messages=[
                ChatMessage.system(system_prompt),
                ChatMessage.user(f'"""{task}"""'),
                *messages,
                final_instruction_msg,
            ],
            prefill_response=response_prefill,
        )

    def build_system_prompt(
        self,
        ai_profile: AIProfile,
        ai_directives: AIDirectives,
        commands: Iterable[Command],
    ) -> tuple[str, str]:
        """
        Builds the system prompt.

        Returns:
            str: The system prompt body
            str: The desired start for the LLM's response; used to steer the output
        """
        response_fmt_instruction, response_prefill = self.response_format_instruction()
        system_prompt_parts = (
            self._generate_intro_prompt(ai_profile)
            + [
                "## Your Task\n"
                "The user will specify a task for you to execute, in triple quotes,"
                " in the next message. Your job is to complete the task, "
                "and terminate when your task is done."
            ]
            + ["## Available Functions\n" + self._generate_function_headers(commands)]
            + ["## RESPONSE FORMAT\n" + response_fmt_instruction]
        )

        # Join non-empty parts together into paragraph format
        return (
            "\n\n".join(filter(None, system_prompt_parts)).strip("\n"),
            response_prefill,
        )

    def response_format_instruction(self) -> tuple[str, str]:
        response_schema = self.response_schema.copy(deep=True)

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface(_RESPONSE_INTERFACE_NAME),
        )
        response_prefill = f'{{\n    "{list(response_schema.properties.keys())[0]}":'

        return (
            (
                f"YOU MUST ALWAYS RESPOND WITH A JSON OBJECT OF THE FOLLOWING TYPE:\n"
                f"{response_format}"
            ),
            response_prefill,
        )

    def _generate_intro_prompt(self, ai_profile: AIProfile) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}.",
            # "Your decisions must always be made independently without seeking "
            # "user assistance. Play to your strengths as an LLM and pursue "
            # "simple strategies with no legal complications.",
        ]

    def _generate_function_headers(self, commands: Iterable[Command]) -> str:
        return "\n\n".join(
            f.header
            + "\n"
            + indent(
                (
                    '"""\n'
                    f"{f.description}\n\n"
                    "Params:\n"
                    + indent(
                        "\n".join(
                            f"{param.name}: {param.spec.description}"
                            for param in f.parameters
                            if param.spec.description
                        )
                    )
                    + "\n"
                    '"""\n'
                    "pass"
                ),
            )
            for f in commands
        )

    async def parse_response_content(
        self,
        response: AssistantChatMessage,
    ) -> OneShotAgentActionProposal:
        if not response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        self.logger.debug(
            "LLM response content:"
            + (
                f"\n{response.content}"
                if "\n" in response.content
                else f" '{response.content}'"
            )
        )
        assistant_reply_dict = extract_dict_from_json(response.content)

        parsed_response = CodeFlowAgentActionProposal.parse_obj(assistant_reply_dict)
        if not parsed_response.python_code:
            raise ValueError("python_code is empty")

        available_functions = {
            c.name: FunctionDef(
                name=c.name,
                arg_types=[(p.name, p.spec.python_type) for p in c.parameters],
                arg_descs={p.name: p.spec.description for p in c.parameters},
                arg_defaults={
                    p.name: p.spec.default or "None"
                    for p in c.parameters
                    if p.spec.default or not p.spec.required
                },
                return_type=c.return_type,
                return_desc="Output of the function",
                function_desc=c.description,
                is_async=c.is_async,
            )
            for c in self.commands
        }
        available_functions.update(
            {
                "main": FunctionDef(
                    name="main",
                    arg_types=[],
                    arg_descs={},
                    return_type="str",
                    return_desc="Output of the function",
                    function_desc="The main function to execute the plan",
                    is_async=True,
                )
            }
        )
        code_validation = await CodeValidator(
            function_name="main",
            available_functions=available_functions,
        ).validate_code(parsed_response.python_code)

        # TODO: prevent combining finish with other functions
        if re.search(r"finish\((.*?)\)", code_validation.functionCode):
            finish_reason = re.search(
                r"finish\((reason=)?(.*?)\)", code_validation.functionCode
            ).group(2)
            result = OneShotAgentActionProposal(
                thoughts=parsed_response.thoughts,
                use_tool=AssistantFunctionCall(
                    name="finish",
                    arguments={"reason": finish_reason[1:-1]},
                ),
            )
        else:
            result = OneShotAgentActionProposal(
                thoughts=parsed_response.thoughts,
                use_tool=AssistantFunctionCall(
                    name="execute_code_flow",
                    arguments={
                        "python_code": code_validation.functionCode,
                        "plan_text": parsed_response.immediate_plan,
                    },
                ),
            )
        return result
