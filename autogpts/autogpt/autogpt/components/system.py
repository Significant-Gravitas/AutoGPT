import platform
import re
from typing import Iterator

import distro

from autogpt.command_decorator import command
from autogpt.agents.utils.exceptions import (
    AgentFinished,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.agents.components import (
    Component,
)
from autogpt.agents.protocols import CommandProvider, MessageProvider
from autogpt.models.command import Command
from autogpts.autogpt.autogpt.agents.agent import DEFAULT_RESPONSE_SCHEMA, AgentSettings
from autogpts.autogpt.autogpt.config.config import Config
from autogpts.autogpt.autogpt.core.resource.model_providers.schema import (
    ChatMessage,
)
from autogpts.autogpt.autogpt.prompts.utils import format_numbered_list


class SystemComponent(Component, MessageProvider, CommandProvider):
    def __init__(self, config: Config, settings: AgentSettings, agent):
        self.legacy_config = config
        self.settings = settings
        #TODO kcze temp to add commands to the messages
        self.agent = agent

    def get_messages(self) -> Iterator[ChatMessage]:
        ai_profile = self.settings.ai_profile
        ai_directives = self.settings.directives
        task = self.settings.task
        #TODO kcze!
        use_functions_api = False

        yield ChatMessage.system(
            f"You are {ai_profile.ai_name}, {ai_profile.ai_role.rstrip('.')}."
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications."
            f"{self._generate_os_info()}\n"
            "## Constraints\n"
            "You operate within the following constraints:\n"
            f"{format_numbered_list( 
                ai_directives.constraints 
                + self._generate_budget_constraint(ai_profile.api_budget)
            )}\n"
            "## Resources\n"
            "You can leverage access to the following resources:\n"
            f"{format_numbered_list(ai_directives.resources)}\n"
            "## Commands\n"
            "These are the ONLY commands you can use."
            " Any action you perform must be possible through one of these commands:\n"
            f"{format_numbered_list([str(cmd) for cmd in self.agent.commands])}\n"
            "## Best practices\n"
            f"{format_numbered_list(ai_directives.best_practices)}\n"
            "## Your Task\n"
            "The user will specify a task for you to execute, in triple quotes,"
            " in the next message. Your job is to complete the task while following"
            " your directives as given above, and terminate when your task is done."
        )
        yield ChatMessage.user(f'"""{task}"""')
        yield ChatMessage.system(self._response_format_instruction(use_functions_api))

    def get_commands(self) -> Iterator[Command]:
        yield self.finish.command
    
    def _generate_os_info(self) -> str:
        """Generates the OS information part of the prompt."""
        if not self.legacy_config.execute_local_commands:
            return ""
        
        os_name = platform.system()
        os_info = (
            platform.platform(terse=True)
            if os_name != "Linux"
            else distro.name(pretty=True)
        )
        return f"The OS you are running on is: {os_info}"
    
    def _generate_budget_constraint(self, api_budget: float) ->list[str]:
        """Generates the budget information part of the prompt."""
        if api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${api_budget:.3f}"
            ]
        return []
    
    def _response_format_instruction(self, use_functions_api: bool) -> str:
        response_schema = DEFAULT_RESPONSE_SCHEMA.copy(deep=True)
        if (
            use_functions_api
            and response_schema.properties
            and "command" in response_schema.properties
        ):
            del response_schema.properties["command"]

        # Unindent for performance
        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        instruction = (
            "Respond with pure JSON containing your thoughts, " "and invoke a tool."
            if use_functions_api
            else "Respond with pure JSON."
        )

        return (
            f"{instruction} "
            "The JSON object should be compatible with the TypeScript type `Response` "
            f"from the following:\n{response_format}"
        )

    @command(
        parameters={
            "reason": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="A summary to the user of how the goals were accomplished",
                required=True,
            ),
        }
    )
    def finish(self, reason: str):
        """Use this to shut down once you have completed your task,
        or when there are insurmountable problems that make it impossible
        for you to finish your task."""
        raise AgentFinished(reason)
