"""Main script for the autogpt package."""
import logging
import os
from pathlib import Path
from colorama import Fore
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments

from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory

from autogpt.prompts.prompt import construct_prompt
from autogpt.plugins import load_plugins


# Load environment variables from .env file


def main() -> None:
    """Main function for the script"""
    cfg = Config()
    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    plugins_found = load_plugins(Path(os.getcwd()) / "plugins")
    loaded_plugins = []
    for plugin in plugins_found:
        if plugin.__name__ in cfg.plugins_blacklist:
            continue
        if plugin.__name__ in cfg.plugins_whitelist:
            loaded_plugins.append(plugin())
        else:
            ack = input(
                f"WARNNG Plugin {plugin.__name__} found. But not in the"
                " whitelist... Load? (y/n): "
            )
            if ack.lower() == "y":
                loaded_plugins.append(plugin())

    if loaded_plugins:
        print(f"\nPlugins found: {len(loaded_plugins)}\n" "--------------------")
    for plugin in loaded_plugins:
        print(f"{plugin._name}: {plugin._version} - {plugin._description}")

    cfg.set_plugins(loaded_plugins)

    ai_name = ""
    prompt = construct_prompt()
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
        prompt=prompt,
        user_input=user_input,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
