import pytest

from autogpt.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.config import AIConfig, Config
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
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def writer_agent(agent_test_config, memory_none: NoMemory, workspace: Workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")
    command_registry.import_commands("autogpt.commands.task_statuses")

    ai_config = AIConfig(
        ai_name="write_to_file-GPT",
        ai_role="an AI designed to use the write_to_file command to write 'Hello World' into a file named \"hello_world.txt\" and then use the task_complete command to complete the task.",
        ai_goals=[
            "Use the write_to_file command to write 'Hello World' into a file named \"hello_world.txt\".",
            "Use the task_complete command to complete the task.",
            "Do not use any other commands.",
        ],
    )
    ai_config.command_registry = command_registry

    triggering_prompt = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )
    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="",
        memory=memory_none,
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def memory_management_agent(agent_test_config, memory_json_file, workspace: Workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")
    command_registry.import_commands("autogpt.commands.task_statuses")

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
        ai_name="",
        memory=memory_json_file,
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def get_company_revenue_agent(
    agent_test_config, memory_json_file, workspace: Workspace
):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

    ai_config = AIConfig(
        ai_name="Get-CompanyRevenue",
        ai_role="an autonomous agent that specializes in finding the reported revenue of a company.",
        ai_goals=[
            "Write the revenue of Tesla in 2022 to a file. You should write the number without commas and you should not use signs like B for billion and M for million.",
        ],
    )
    ai_config.command_registry = command_registry

    system_prompt = ai_config.construct_full_prompt()
    Config().set_continuous_mode(False)
    agent = Agent(
        ai_name="Get-CompanyRevenue",
        memory=memory_json_file,
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


@pytest.fixture
def kubernetes_agent(memory_json_file, workspace: Workspace):
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
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent
