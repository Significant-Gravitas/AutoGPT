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
from multigpt.expert import Expert
from multigpt.multi_config import MultiConfig
from multigpt.multi_agent_manager import MultiAgentManager

EXPERT_PROMPT = """The task is: {task}.

Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts and list three goals for them to help the overall task. Follow the format precisely:
1. [Name of the person]: [Description of how they are useful]
1a) [Goal a]
1b) [Goal b]
1c) [Goal c]"""

EXPERT_TRAITS_PROMPT = """
    Rate {name} on a scale from 0 (extremly low degree of) to 10 (extremly high degree of) on the following five traits: Openness, Agreeableness, Conscientiousness, Emotional Stability and Assertiveness. Follow the format precisely:
    [Name of Person]
    Openness: [0-10]
    Agreeableness: [0-10]
    Conscientiousness: [0-10]
    Emotional Stability: [0-10]
    Assertiveness: [0-10]
    [Short description of personality traits of {name}]
"""


def parse_experts(experts: str) -> List[Expert]:
    # personas = experts.split(r"[0-9]\. ")
    experts = re.sub("\n", "", experts)
    personas = re.split(r"[0-9]\. ", experts)[1:]
    # print(personas)
    res = []
    for persona in personas:
        try:
            tmp = re.split(r"[0-9][a-c]\) ", persona)
            # print(tmp)
            name, description = tmp[0].split(":")[:2]
            # print(name, description)
            goals = tmp[1:]
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
            # print(name, description, goals)
            res.append(Expert(name, description, goals))
        except:
            print("Error parsing expert")
    # TODO: assert res length is not larger than MAX_EXPERTS
    return res


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

    messages = [{"role": "system", "content": EXPERT_PROMPT.format(task=task, min_experts=cfg.min_experts,
                                                                   max_experts=cfg.max_experts)}]
    with Spinner("Gathering group of experts... "):
        experts_string = create_chat_completion(messages=messages, model=cfg.smart_llm_model, max_tokens=1000)

    experts = parse_experts(experts_string)
    multi_agent_manager.set_experts(experts)
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)

    for expert in experts:
        messages = [{"role": "system", "content": EXPERT_TRAITS_PROMPT.format(name=expert.ai_name)}]
        with Spinner(f"Generating trait profile for {expert.ai_name}... "):
            expert_traits = create_chat_completion(messages=messages, model=cfg.smart_llm_model, max_tokens=1000)
        expert.set_traits(expert_traits)
        logger.typewriter_log(
            "\nTrait profile:", Fore.RED,
            expert_traits, speak_text=True
        )
        multi_agent_manager.create_agent(expert)

    multi_agent_manager.start_interaction_loop()


if __name__ == "__main__":
    main()
