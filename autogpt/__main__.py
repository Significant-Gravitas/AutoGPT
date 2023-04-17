"""Main script for the autogpt package."""
import logging
from colorama import Fore
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments
from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.prompt import construct_prompt
# Load environment variables from .env file


def main() -> None:
    """Main function for the script"""
    cfg = Config()
    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    ai_name = ""
    system_prompt = construct_prompt()
    # print(prompt)
    # Initialize variables
    full_message_history = []
    next_action_count = 0
    # Make a constant:
    triggering_prompt = (
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
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
