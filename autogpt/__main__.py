"""Main script for the autogpt package."""
import logging
import os
from pathlib import Path
from colorama import Fore
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments
from autogpt.commands.command import CommandRegistry
from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory

from autogpt.prompts.prompt import construct_main_ai_config
from autogpt.plugins import load_plugins


# Load environment variables from .env file


def main() -> None:
    """Main function for the script"""
    cfg = Config()
    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    cfg.set_plugins(load_plugins(cfg, cfg.debug_mode))
    # Create a CommandRegistry instance and scan default folder
    command_registry = CommandRegistry()
    command_registry.import_commands("scripts.ai_functions")
    command_registry.import_commands("scripts.commands")
    command_registry.import_commands("scripts.execute_code")
    command_registry.import_commands("scripts.agent_manager")
    command_registry.import_commands("scripts.file_operations")
    ai_name = ""
    ai_config = construct_main_ai_config()
    # print(prompt)
    # Initialize variables
    full_message_history = []
    next_action_count = 0
    # Make a constant:
    user_input = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )
    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        command_registry=command_registry,
        config=ai_config,
        prompt=ai_config.construct_full_prompt(),
        user_input=user_input,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
