import logging

from colorama import Fore

from autogpt.agent import Agent
from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory
from autogpt.prompt import Prompt


def create_agent(prompt: Prompt) -> Agent:
    cfg = Config()
    # TODO: fill in llm values here
    check_openai_api_key()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    ai_name = ""
    # print(prompt)
    # Initialize variables
    full_message_history = []
    next_action_count = 0
    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(
        "Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
    )
    logger.typewriter_log("Using Browser:", Fore.GREEN, cfg.selenium_web_browser)

    return Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        system_prompt=prompt.system_prompt,
        triggering_prompt=prompt.triggering_prompt,
    )
