"""Main script for the autogpt package."""
import logging

from colorama import Fore

from autogpt.agent import Agent
from autogpt.args import parse_arguments
from autogpt.config import AIConfig, Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.prompt import construct_full_prompt
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

# Load environment variables from .env file


def get_ai_config(cfg: Config) -> str:
    """
    Get AIConfig class object that contains the configuration information for the AI.

    Parameters:
        cfg (Config): System configuration class object.

    Returns:
        str: The prompt string
    """
    ai_config = AIConfig.load(cfg.ai_settings_file)
    if cfg.skip_reprompt and ai_config.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, ai_config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, ai_config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{ai_config.ai_goals}")
    elif ai_config.ai_name:
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {ai_config.ai_name}?",
            speak_text=True,
        )
        should_continue = clean_input(
            f"""Continue with the last settings?
Name:  {ai_config.ai_name}
Role:  {ai_config.ai_role}
Goals: {ai_config.ai_goals}
Continue (y/n): """
        )
        should_continue_command = should_continue.lower().strip()
        if should_continue_command == "n":
            ai_config = AIConfig()

    if not ai_config.ai_name:
        ai_config = prompt_user()
        ai_config.save(cfg.ai_settings_file)

    return ai_config


def main() -> None:
    """Main function for the script"""
    cfg = Config()

    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()

    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)

    ai_config = get_ai_config(cfg)

    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    # Initialize variables
    full_message_history = []
    next_action_count = 0

    system_prompt = construct_full_prompt(cfg, ai_config)
    # print(system_prompt)
    # Make a constant:
    triggering_prompt = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )

    agent = Agent(
        ai_name=ai_config.ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
