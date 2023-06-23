"""Set up the AI and its goals"""
import re
from typing import Optional

from colorama import Fore
from jinja2 import Template

from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.llm.base import ChatSequence, Message
from autogpt.llm.chat import create_chat_completion
from autogpt.logs import logger
from autogpt.prompts.default_prompts import (
    DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC,
    DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC,
)


def generate_aiconfig_automatic(config: Config, user_prompt: str) -> Optional[AIConfig]:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """
    from autogpt.prompts.prompt import handle_config

    system_prompt = DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC
    prompt_ai_config_automatic = Template(
        DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC
    ).render(user_prompt=user_prompt)

    # Call LLM with the string as user input
    output = create_chat_completion(
        ChatSequence.for_model(
            config.fast_llm_model,
            [
                Message("system", system_prompt),
                Message("user", prompt_ai_config_automatic),
            ],
        ),
        config,
    ).content

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    match = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE)
    ai_name = match.group(1) if match is not None else ""

    match = re.search(
        r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    ai_role = match.group(1).strip() if match is not None else ""

    ai_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
    api_budget = 0.0  # TODO: parse api budget using a regular expression

    if config.plugins_allowlist:
        plugins = config.plugins_allowlist
    else:
        plugins = []

    # Fallback to manual when automatic generation failed
    if not ai_name or not ai_role or not ai_goals:
        logger.typewriter_log(
            "Automatic generation failed. Falling back to manual generation...",
            Fore.RED,
        )
        return handle_config(config, None, "create")

    # Create new AIConfig instance with parsed parameters
    new_config = AIConfig(ai_name, ai_role, ai_goals, api_budget, plugins)

    # Save the new configuration
    try:
        new_config.save(config.ai_settings_filepath, append=True)
    except ValueError as e:
        logger.typewriter_log(f"An error occurred: {e}", Fore.RED)
    else:
        logger.typewriter_log("Configuration saved.", Fore.GREEN)

    return new_config
