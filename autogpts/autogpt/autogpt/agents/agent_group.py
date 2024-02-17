import logging
from typing import Optional
from autogpt.agents.prompt_strategies.divide_and_conquer import (
    DivideAndConquerAgentPromptStrategy,
)
from autogpt.file_storage import get_storage
from autogpt.file_storage.base import FileStorage
from autogpt.models.command_registry import CommandRegistry
from forge.sdk.model import TaskRequestBody
from autogpt.config.config import Config, ConfigBuilder
from autogpt.agents.agent import Agent, AgentConfiguration
from autogpt.agent_manager.agent_manager import AgentManager
from autogpt.agents.agent_member import AgentMember, AgentMemberSettings, AgentTask, AgentTaskSettings
from autogpt.agent_factory.profile_generator import AgentProfileGenerator
from autogpt.core.resource.model_providers.schema import ChatModelProvider

logger = logging.getLogger(__name__)


class AgentGroup:

    leader: AgentMember
    members: dict[str, AgentMember]

    def assign_group_to_members(self):
        self.leader.recursive_assign_group(self)

    def reload_members(self):
        members = self.leader.get_list_of_all_your_team_members()
        members_dict = {}
        for agent_member in members:
            members_dict[agent_member.id] = agent_member
        self.members = members_dict

    def __init__(self, leader: AgentMember):
        self.leader = leader
        self.assign_group_to_members()
        self.reload_members()

    async def create_task(self, task: TaskRequestBody):
        await self.leader.create_task(task)

    @staticmethod
    def configure_agent_group_with_state(
        state: AgentMemberSettings,
        app_config: Config,
        file_storage: FileStorage,
        llm_provider: ChatModelProvider,
    ) -> 'AgentGroup':
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
        if state.create_agent:
            commands.insert(0, "autogpt.commands.create_agent")
        else:
            commands.insert(0, "autogpt.commands.request_agent")

        leader = AgentMember(
            settings=state,
            llm_provider=llm_provider,
        )
        agents, tasks = AgentGroup.create_agents_and_tasks_dict_from_state(
            leader, file_storage, llm_provider
        )
        members = AgentGroup.get_agents_tree_from_state(agents, tasks, state)
        if state.recruiter_id:
            leader.recruiter = agents[state.recruiter_id]
        leader.members = members
        return AgentGroup(leader=leader)

    @staticmethod
    def get_agents_tree_from_state(
        agents: dict[str, AgentMember], tasks: dict[str, AgentTask], agent: AgentMember
    ):
        members = []
        for member_id in agent.state.members:
            members_of_member = AgentGroup.get_agents_and_tasks_from_state(
                agents, tasks, agents[member_id]
            )
            agents[member_id].members = members_of_member
            agent_tasks = []
            for task_id in agent.state.tasks:
                if tasks[task_id].parent_task_id:
                    tasks[task_id].parent_task = tasks[tasks[task_id].parent_task_id]
                agent_tasks.append(
                    tasks[task_id]
                )
            agents[member_id].tasks = agent_tasks
            members.append(agents[member_id])
        return members

    @staticmethod
    def create_agents_and_tasks_dict_from_state(
        agent: "AgentMember", file_storage: FileStorage, llm_provider: ChatModelProvider
    ):
        agent_manager = AgentManager(file_storage)
        members = []
        sub_tasks = []
        for member_id in agent.state.members:
            member_state = agent_manager.load_agent_state(member_id)
            agent_member = AgentMember(
                settings=member_state,
                llm_provider=llm_provider,
            )
            members, sub_tasks = AgentGroup.get_agents_and_tasks_from_state(
                agent_member, file_storage, llm_provider
            )
            members[agent_member.id] = agent_member
            for sub_task in agent.state.tasks:
                sub_tasks[sub_task.task_id] = AgentTask(
                    task_id=sub_task.task_id,
                    input=sub_task.input,
                    parent_task_id=sub_task.parent_task_id,
                )
        return members, sub_tasks


async def create_agent_member(
    role: str,
    initial_prompt: str,
    llm_provider: ChatModelProvider,
    boss: Optional["AgentMember"] = None,
    recruiter: Optional["AgentMember"] = None,
    create_agent: bool = False,
) -> AgentMember:
    config = ConfigBuilder.build_config_from_env()
    config.logging.plain_console_output = True

    config.continuous_mode = False
    config.continuous_limit = 20
    config.noninteractive_mode = True
    config.memory_backend = "no_memory"
    settings = await generate_agent_settings_for_task(
        role=role,
        initial_prompt=initial_prompt,
        boss_id=boss.id if boss else None,
        recruiter_id=recruiter.id if recruiter else None,
        tasks=[],
        members=[],
        create_agent=create_agent,
        task=initial_prompt,
        app_config=config,
        llm_provider=llm_provider,
    )

    agent_member = AgentMember(
        settings=settings,
        boss=boss,
        recruiter=recruiter,
        llm_provider=llm_provider,
    )

    if boss:
        boss.members.append(agent_member)
        boss.group.reload_members()

    await agent_member.save_state()
    return agent_member


async def generate_agent_settings_for_task(
    role: str,
    initial_prompt: str,
    boss_id: Optional[str],
    recruiter_id: Optional[str],
    tasks: list[AgentTaskSettings],
    members: list[str],
    create_agent: bool,
    task: str,
    llm_provider: ChatModelProvider,
    app_config,
) -> AgentMemberSettings:
    agent_profile_generator = AgentProfileGenerator(
        **AgentProfileGenerator.default_configuration.dict()  # HACK
    )

    prompt = agent_profile_generator.build_prompt(task)
    output = (
        await llm_provider.create_chat_completion(
            prompt.messages,
            model_name=app_config.smart_llm,
            functions=prompt.functions,
        )
    ).response

    ai_profile, ai_directives = agent_profile_generator.parse_response_content(output)

    return AgentMemberSettings(
        role=role,
        initial_prompt=initial_prompt,
        boss_id=boss_id,
        recruiter_id=recruiter_id,
        tasks=tasks,
        members=members,
        create_agent=create_agent,
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        task=task,
        ai_profile=ai_profile,
        directives=ai_directives,
        config=AgentConfiguration(
            fast_llm=app_config.fast_llm,
            smart_llm=app_config.smart_llm,
            allow_fs_access=not app_config.restrict_to_workspace,
            use_functions_api=app_config.openai_functions,
            plugins=app_config.plugins,
        ),
        history=Agent.default_settings.history.copy(deep=True),
    )
