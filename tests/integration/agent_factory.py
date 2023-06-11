import pytest

from autogpt.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.config import AIConfig, Config
from autogpt.main import COMMAND_CATEGORIES
from autogpt.memory.vector import NoMemory, get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT
from autogpt.workspace import Workspace


@pytest.fixture
def agent_test_config(config: Config):
    was_continuous_mode = config.continuous_mode
    was_temperature = config.temperature
    was_plain_output = config.plain_output
    config.set_continuous_mode(False)
    config.set_temperature(0)
    config.plain_output = True
    yield config
    config.set_continuous_mode(was_continuous_mode)
    config.set_temperature(was_temperature)
    config.plain_output = was_plain_output


@pytest.fixture
def memory_json_file(agent_test_config: Config):
    was_memory_backend = agent_test_config.memory_backend

    agent_test_config.set_memory_backend("json_file")
    yield get_memory(agent_test_config, init=True)

    agent_test_config.set_memory_backend(was_memory_backend)


@pytest.fixture
def browser_agent(agent_test_config, memory_none: NoMemory, workspace: Workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.commands.web_selenium")
    command_registry.import_commands("autogpt.app")
    command_registry.import_commands("autogpt.commands.task_statuses")

    ai_config = AIConfig(
        ai_name="browse_website-GPT",
        ai_role="an AI designed to use the browse_website command to visit http://books.toscrape.com/catalogue/meditations_33/index.html, answer the question 'What is the price of the book?' and write the price to a file named \"browse_website.txt\", and use the task_complete command to complete the task.",
        ai_goals=[
            "Use the browse_website command to visit http://books.toscrape.com/catalogue/meditations_33/index.html and answer the question 'What is the price of the book?'",
            'Write the price of the book to a file named "browse_website.txt".',
            "Use the task_complete command to complete the task.",
            "Do not use any other commands.",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="",
        memory=memory_none,
        command_registry=command_registry,
        ai_config=ai_config,
        config=agent_test_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def file_system_agents(
    agent_test_config, memory_json_file: NoMemory, workspace: Workspace
):
    agents = []
    command_registry = get_command_registry(agent_test_config)

    ai_goals = [
        "Write 'Hello World' into a file named \"hello_world.txt\".",
        'Write \'Hello World\' into 2 files named "hello_world_1.txt"and "hello_world_2.txt".',
    ]

    for ai_goal in ai_goals:
        ai_config = AIConfig(
            ai_name="File System Agent",
            ai_role="an AI designed to manage a file system.",
            ai_goals=[ai_goal],
        )
        ai_config.command_registry = command_registry
        system_prompt = ai_config.construct_full_prompt()
        Config().set_continuous_mode(False)
        agents.append(
            Agent(
                ai_name="File System Agent",
                memory=memory_json_file,
                command_registry=command_registry,
                ai_config=ai_config,
                config=agent_test_config,
                next_action_count=0,
                system_prompt=system_prompt,
                triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
                workspace_directory=workspace.root,
            )
        )
    return agents


@pytest.fixture
def memory_management_agent(agent_test_config, memory_json_file, workspace: Workspace):
    command_registry = get_command_registry(agent_test_config)

    ai_config = AIConfig(
        ai_name="Follow-Instructions-GPT",
        ai_role="an AI designed to read the instructions_1.txt file using the read_file method and follow the instructions in the file.",
        ai_goals=[
            "Use the command read_file to read the instructions_1.txt file",
            "Follow the instructions in the instructions_1.txt file",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="Follow-Instructions-GPT",
        memory=memory_json_file,
        command_registry=command_registry,
        ai_config=ai_config,
        config=agent_test_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def information_retrieval_agents(
    agent_test_config, memory_json_file, workspace: Workspace
):
    agents = []
    command_registry = get_command_registry(agent_test_config)

    ai_goals = [
        "Write to a file called output.txt containing tesla's revenue in 2022 after searching for 'tesla revenue 2022'.",
        "Write to a file called output.txt containing tesla's revenue in 2022.",
        "Write to a file called output.txt containing tesla's revenue every year since its creation.",
    ]
    for ai_goal in ai_goals:
        ai_config = AIConfig(
            ai_name="Information Retrieval Agent",
            ai_role="an autonomous agent that specializes in retrieving information.",
            ai_goals=[ai_goal],
        )
        ai_config.command_registry = command_registry
        system_prompt = ai_config.construct_full_prompt()
        Config().set_continuous_mode(False)
        agents.append(
            Agent(
                ai_name="Information Retrieval Agent",
                memory=memory_json_file,
                command_registry=command_registry,
                ai_config=ai_config,
                config=agent_test_config,
                next_action_count=0,
                system_prompt=system_prompt,
                triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
                workspace_directory=workspace.root,
            )
        )
    return agents


@pytest.fixture
def kubernetes_agent(
    agent_test_config: Config, memory_json_file: NoMemory, workspace: Workspace
) -> Agent:
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

    ai_config = AIConfig(
        ai_name="Kubernetes",
        ai_role="an autonomous agent that specializes in creating Kubernetes deployment templates.",
        ai_goals=[
            "Write a simple kubernetes deployment file and save it as a kube.yaml.",
            # You should make a simple nginx web server that uses docker and exposes the port 80.
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()
    Config().set_continuous_mode(False)
    agent = Agent(
        ai_name="Kubernetes-Demo",
        memory=memory_json_file,
        command_registry=command_registry,
        ai_config=ai_config,
        config=agent_test_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def get_nobel_prize_agent(agent_test_config, memory_json_file, workspace: Workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")
    command_registry.import_commands("autogpt.commands.web_selenium")

    ai_config = AIConfig(
        ai_name="Get-PhysicsNobelPrize",
        ai_role="An autonomous agent that specializes in physics history.",
        ai_goals=[
            "Write to file the winner's name(s), affiliated university, and discovery of the 2010 nobel prize in physics. Write your final answer to 2010_nobel_prize_winners.txt.",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()
    Config().set_continuous_mode(False)

    agent = Agent(
        ai_name="Get-PhysicsNobelPrize",
        memory=memory_json_file,
        command_registry=command_registry,
        ai_config=ai_config,
        config=agent_test_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def debug_code_agents(agent_test_config, memory_json_file, workspace: Workspace):
    agents = []
    goals = [
        [
            "1- Run test.py using the execute_python_file command.",
            "2- Read code.py using the read_file command.",
            "3- Modify code.py using the write_to_file command."
            "Repeat step 1, 2 and 3 until test.py runs without errors.",
        ],
        [
            "1- Run test.py.",
            "2- Read code.py.",
            "3- Modify code.py."
            "Repeat step 1, 2 and 3 until test.py runs without errors.",
        ],
        ["1- Make test.py run without errors."],
    ]

    for goal in goals:
        ai_config = AIConfig(
            ai_name="Debug Code Agent",
            ai_role="an autonomous agent that specializes in debugging python code",
            ai_goals=goal,
        )
        command_registry = get_command_registry(agent_test_config)
        ai_config.command_registry = command_registry
        system_prompt = ai_config.construct_full_prompt()
        Config().set_continuous_mode(False)
        agents.append(
            Agent(
                ai_name="Debug Code Agent",
                memory=memory_json_file,
                command_registry=command_registry,
                ai_config=ai_config,
                config=agent_test_config,
                next_action_count=0,
                system_prompt=system_prompt,
                triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
                workspace_directory=workspace.root,
            )
        )
    return agents


def get_command_registry(agent_test_config):
    command_registry = CommandRegistry()
    enabled_command_categories = [
        x
        for x in COMMAND_CATEGORIES
        if x not in agent_test_config.disabled_command_categories
    ]
    for command_category in enabled_command_categories:
        command_registry.import_commands(command_category)
    return command_registry
