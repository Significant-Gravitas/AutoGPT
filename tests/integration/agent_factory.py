import os

from autogpt.agent import Agent
from autogpt.app import CFG
from autogpt.commands.command import CommandRegistry
from autogpt.config import AIConfig
from autogpt.memory import get_memory
from autogpt.prompts.prompt import DEFAULT_TRIGGERING_PROMPT


def create_browser_agent(workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.commands.web_selenium")
    command_registry.import_commands("autogpt.app")

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
    CFG.set_continuous_mode(True)
    CFG.set_memory_backend("no_memory")
    CFG.set_temperature(0)

    memory = get_memory(CFG, init=True)
    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="",
        memory=memory,
        full_message_history=[],
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=DEFAULT_TRIGGERING_PROMPT,
        workspace_directory=workspace.root,
    )

    return agent


def create_writer_agent(workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.app")

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
    CFG.set_continuous_mode(True)
    CFG.set_memory_backend("no_memory")
    CFG.set_temperature(0)
    memory = get_memory(CFG, init=True)
    triggering_prompt = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )
    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="",
        memory=memory,
        full_message_history=[],
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
        workspace_directory=workspace.root,
    )

    return agent
