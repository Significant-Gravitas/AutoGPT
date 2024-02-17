import copy
import logging
from enum import Enum
import uuid
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from autogpt.agent_factory.configurators import create_agent
from autogpt.commands import COMMAND_CATEGORIES
from autogpt.config.config import ConfigBuilder
from autogpt.file_storage import get_storage
from forge.sdk.model import Task, TaskRequestBody
from autogpt.agents.base import BaseAgentSettings
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.models.command_registry import CommandRegistry
from autogpt.models.action_history import Action, ActionResult
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.agents.utils.prompt_scratchpad import PromptScratchpad
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from autogpt.core.prompting.schema import (
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatModelProvider,
)
from autogpt.agents.prompt_strategies.divide_and_conquer import (
    CommandRequest,
    DivideAndConquerAgentPromptConfiguration,
    DivideAndConquerAgentPromptStrategy,
)

logger = logging.getLogger(__name__)

class AgentTaskSettings(BaseModel):
    input: str
    task_id: str
    parent_task_id: Optional[str]
    status: str
    sub_tasks: list[str]

class AgentMemberSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: DivideAndConquerAgentPromptConfiguration = Field(
        default_factory=(
            lambda: DivideAndConquerAgentPromptStrategy.default_configuration.copy(
                deep=True
            )
        )
    )
    role: str
    initial_prompt: str
    boss_id: Optional[str]
    recruiter_id: Optional[str]
    create_agent: bool
    members: list[str]
    tasks: list[AgentTaskSettings]

class ProposeActionResult:
    commands: list[CommandRequest]
    agent: "AgentMember"

    def __init__(self, commands: list[CommandRequest], agent: "AgentMember") -> None:
        self.commands = commands
        self.agent = agent


class CommandActionResult:
    action_result: ActionResult
    command: str

    def __init__(self, action_result: ActionResult, command: str) -> None:
        self.action_result = action_result
        self.command = command


class AgentMember(Agent):

    id: str
    role: str
    initial_prompt: str
    boss: Optional["AgentMember"]
    recruiter: Optional["AgentMember"]
    tasks: list["AgentTask"]
    members: list["AgentMember"]
    create_agent: bool
    group: "AgentGroup"

    def recursive_assign_group(self, group: "AgentGroup"):
        self.group = group
        for members in self.members:
            members.recursive_assign_group(group)

    def get_list_of_all_your_team_members(self) -> list["AgentMember"]:
        members = []
        members.append(self)
        for member in self.members:
            members.extend(member.get_list_of_all_your_team_members())
        return members

    def __init__(
        self,
        llm_provider: ChatModelProvider,
        settings: AgentMemberSettings,
        boss: Optional["AgentMember"] = None,
        recruiter: Optional["AgentMember"] = None,
        tasks: list['AgentTask'] = [],
        members: list['AgentMember'] = [],
    ):
        config = ConfigBuilder.build_config_from_env()
        config.logging.plain_console_output = True

        config.continuous_mode = False
        config.continuous_limit = 20
        config.noninteractive_mode = True
        config.memory_backend = "no_memory"

        file_storage = get_storage(
            config.file_storage_backend, root_path="data_group", restrict_to_root=False
        )
        file_storage.initialize()
        commands = [
            "autogpt.commands.create_task",
            "autogpt.commands.execute_code",
            "autogpt.commands.file_operations",
            "autogpt.commands.user_interaction",
            "autogpt.commands.web_search",
            "autogpt.commands.web_selenium",
            "autogpt.commands.finish_task",
            "autogpt.commands.image_gen",
        ]
        if create_agent:
            commands.insert(0, "autogpt.commands.create_agent")
        else:
            commands.insert(0, "autogpt.commands.request_agent")

        command_registry = CommandRegistry.with_command_modules(commands, config)

        hugging_chat_settings = OpenAIProvider.default_settings.copy(deep=True)
        hugging_chat_settings.credentials = config.openai_credentials

        super().__init__(settings, llm_provider, command_registry, file_storage, config)

        self.role = settings.role
        self.initial_prompt = settings.initial_prompt
        self.boss = boss
        self.recruiter = recruiter
        self.tasks = tasks
        self.members = members
        self.create_agent = settings.create_agent
        self.prompt_strategy = DivideAndConquerAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )

    def build_prompt(
        self,
        scratchpad: PromptScratchpad,
        tasks: list["AgentTask"],
        extra_commands: Optional[list[CompletionModelFunction]] = None,
        extra_messages: Optional[list[ChatMessage]] = None,
        **extras,
    ) -> ChatPrompt:
        """Constructs a prompt using `self.prompt_strategy`.

        Params:
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)
            extra_commands: Additional commands that the agent has access to.
            extra_messages: Additional messages to include in the prompt.
        """
        if not extra_commands:
            extra_commands = []
        if not extra_messages:
            extra_messages = []

        # Apply additions from plugins
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(scratchpad)
        ai_directives = self.directives.copy(deep=True)
        ai_directives.resources += scratchpad.resources
        ai_directives.constraints += scratchpad.constraints
        ai_directives.best_practices += scratchpad.best_practices
        extra_commands += list(scratchpad.commands.values())

        prompt = self.prompt_strategy.build_prompt(
            include_os_info=True,
            tasks=tasks,
            agent_member=self,
            ai_profile=self.ai_profile,
            ai_directives=ai_directives,
            commands=get_openai_command_specs(
                self.command_registry.list_available_commands(self)
            )
            + extra_commands,
            event_history=self.event_history,
            max_prompt_tokens=self.send_token_limit,
            count_tokens=lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            count_message_tokens=lambda x: self.llm_provider.count_message_tokens(
                x, self.llm.name
            ),
            extra_messages=extra_messages,
            **extras,
        )

        return prompt

    async def execute_commands(
        self, commands: list["CommandRequest"]
    ) -> list[CommandActionResult]:
        results = []

        for command in commands:
            self.event_history.register_action(
                Action(
                    name=command.command,
                    args=command.args,
                    reasoning="",
                )
            )
            result = await self.execute(
                command_name=command.command,
                command_args=command.args,
                user_input=command.user_input,
            )
            results.append(
                CommandActionResult(action_result=result, command=command.command)
            )

        return results

    async def create_task(self, task_request: TaskRequestBody):
        try:
            task = AgentTask(
                input=task_request.input,
                additional_input=task_request.additional_input,
                status=AgentTaskStatus.INITIAL.value,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                task_id=str(uuid.uuid4()),
                sub_tasks=[],
            )

            self.tasks.append(task)
        except Exception as e:
            logger.error(f"Error occurred while creating task: {e}")

    async def recursive_propose_action(self) -> list[ProposeActionResult]:
        result = [
            ProposeActionResult(agent=self, commands=await self.single_propose_action())
        ]
        for agent_member in self.members:
            result = result + (await agent_member.recursive_propose_action())
        return result

    async def single_propose_action(self) -> list[CommandRequest]:
        current_tasks = []
        for task in self.tasks:
            if task.status == AgentTaskStatus.REJECTED:
                task.status = AgentTaskStatus.INITIAL

            elif task.status == AgentTaskStatus.DOING:
                # sub_tasks_done = all(sub_task.status == AgentTaskStatus.DONE for sub_task in task.sub_tasks)
                # # sub_tasks_checking = any(sub_task.status == AgentTaskStatus.CHECKING for sub_task in task.sub_tasks)

                # if sub_tasks_done:
                #     task.status = AgentTaskStatus.CHECKING
                # elif sub_tasks_checking:
                current_tasks.append(task)

            elif task.status == AgentTaskStatus.INITIAL:
                current_tasks.append(task)
                task.status = AgentTaskStatus.DOING

        commands: list[CommandRequest] = []
        if current_tasks:
            self._prompt_scratchpad = PromptScratchpad()
            logger.info(f"tasks: {str(current_tasks)}")
            prompt = self.build_prompt(
                scratchpad=self._prompt_scratchpad, tasks=current_tasks
            )
            result = await self.llm_provider.create_chat_completion(
                prompt.messages,
                model_name=self.config.smart_llm,
                functions=prompt.functions,
                completion_parser=lambda r: self.parse_and_process_response(
                    r,
                    prompt,
                    scratchpad=self._prompt_scratchpad,
                ),
            )
            commands = result.parsed_result
        await self.save_state()
        return commands

    def parse_and_process_response(
        self, llm_response: AssistantChatMessage, *args, **kwargs
    ) -> list[CommandRequest]:
        result = self.prompt_strategy.parse_response_content(llm_response)
        return result


class AgentTaskStatus(Enum):
    INITIAL = "INITIAL"
    DOING = "DOING"
    CHECKING = "CHECKING"
    REJECTED = "REJECTED"
    DONE = "DONE"


class AgentTask(Task):
    parent_task_id: Optional[str]
    status: AgentTaskStatus
    parent_task: Optional["AgentTask"]
    sub_tasks: list["AgentTask"]
