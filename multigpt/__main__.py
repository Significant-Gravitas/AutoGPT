"""Main script for the autogpt package."""
import logging
import os
import re
from typing import List
from colorama import Fore
from slugify import slugify

from autogpt import utils
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments

from autogpt.config import check_openai_api_key
from autogpt.llm_utils import create_chat_completion
from autogpt.logs import logger
from autogpt.memory import get_memory

from autogpt.spinner import Spinner
from multigpt import lmql_utils
from multigpt.agent_traits import AgentTraits
from multigpt.expert import Expert
from multigpt.multi_config import MultiConfig
from multigpt.multi_agent_manager import MultiAgentManager


def main() -> None:
    cfg = MultiConfig()
    check_openai_api_key()
    parse_arguments()
    multi_agent_manager = MultiAgentManager(cfg)

    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    ai_name = "AI Orchestrator"

    logger.typewriter_log(
        "Welcome to MultiGPT!", Fore.BLUE,
        "I am the orchestrator of your AI assistants.", speak_text=True
    )

    logger.typewriter_log(
        "Define the task you want to accomplish and I will gather a group of expertGPTs to help you.", Fore.BLUE,
        "Be specific. Prefer 'Achieve world domination by creating a raccoon army!' to 'Achieve world domination!'",
        speak_text=True,
    )

    task = utils.clean_input("Task: ")
    if task == "":
        task = "Achieve world domination!"

    experts = []
    with Spinner("Gathering group of experts... "):
        experts = lmql_utils.lmql_generate_experts(task=task, min_experts=cfg.min_experts, max_experts=cfg.max_experts)

    multi_agent_manager.set_experts(experts)
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)

    for name, description, goals in experts:

        with Spinner(f"Generating trait profile for {name}... "):
            traits_list = list(lmql_utils.lmql_generate_trait_profile(name).values())
            expert_traits = AgentTraits(*traits_list)

        expert = Expert(name, description, goals, expert_traits)
        logger.typewriter_log(
            f"{name}", Fore.BLUE,
            f"{description}", speak_text=True
        )
        goals_str = ""
        for i, goal in enumerate(goals):
            goals_str += f"{i + 1}. {goal}\n"
        logger.typewriter_log(
            f"Goals:", Fore.GREEN, goals_str
        )
        logger.typewriter_log(
            "\nTrait profile:", Fore.RED,
            str(expert_traits), speak_text=True
        )
        multi_agent_manager.create_agent(expert)

    multi_agent_manager.start_interaction_loop()


if __name__ == "__main__":
    main()
