"""Setup the AI and its goals"""
import glob
import os
from typing import List

from colorama import Fore, Style
from autogpt import utils
from autogpt.spinner import Spinner
from multigpt import lmql_utils
from multigpt.agent_traits import AgentTraits
from multigpt.expert import Expert
from autogpt.logs import logger


def prompt_user(cfg, multi_agent_manager):
    logger.typewriter_log(
        "Welcome to MultiGPT!", Fore.BLUE,
        "I am the orchestrator of your AI assistants.", speak_text=True
    )
    experts = []
    saved_agents_directory = os.path.join(os.path.dirname(__file__), "saved_agents")
    if os.path.exists(saved_agents_directory):
        file_pattern = os.path.join(saved_agents_directory, '*.yaml')
        yaml_files = glob.glob(file_pattern)
        if yaml_files:
            agent_names = []
            for yaml_file in yaml_files:
                expert = Expert.load(yaml_file)
                agent_names.append(expert.ai_name)
                experts.append(expert)
            logger.typewriter_log(
                "Found existing agents!", Fore.BLUE,
                f"List of agents: {agent_names}", speak_text=True
            )
            loading = utils.clean_input("Do you want me to load these agents [Y/n]: ")
            if loading.upper() == "Y":

                logger.typewriter_log(
                    f"LOADING SUCCESSFUL!", Fore.YELLOW
                )

                for expert in experts:
                    logger.typewriter_log(
                        f"{expert.ai_name}", Fore.BLUE,
                        f"{expert.ai_role}", speak_text=True
                    )
                    goals_str = ""
                    for i, goal in enumerate(expert.ai_goals):
                        goals_str += f"{i + 1}. {goal}\n"
                    logger.typewriter_log(
                        f"Goals:", Fore.GREEN, goals_str
                    )
                    logger.typewriter_log(
                        "\nTrait profile:", Fore.RED,
                        str(expert.ai_traits), speak_text=True
                    )
                additional_agents = utils.clean_input("Do you want to create additional agents with a new task that join the discussion? [Y/n]: ")
                if additional_agents.upper() == "Y":
                    pass
                else:
                    for expert in experts:
                        multi_agent_manager.create_agent(expert)
                    return
            elif loading.upper() == "N":
                pass
            else:
                exit(1)

    logger.typewriter_log(
        "Define the task you want to accomplish and I will gather a group of expertGPTs to help you.", Fore.BLUE,
        "Be specific. Prefer 'Achieve world domination by creating a raccoon army!' to 'Achieve world domination!'",
        speak_text=True,
    )

    task = utils.clean_input("Task: ")
    if task == "":
        task = "Achieve world domination!"

    with Spinner("Gathering group of experts... "):
        experts_parsed = lmql_utils.lmql_generate_experts(task=task, min_experts=cfg.min_experts, max_experts=cfg.max_experts)

    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    for name, description, goals in experts_parsed:

        with Spinner(f"Generating trait profile for {name}... "):
            traits_list = list(lmql_utils.lmql_generate_trait_profile(name).values())
            expert_traits = AgentTraits(*traits_list)

        experts.append(Expert(name, description, goals, expert_traits))
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

    for expert in experts:
        multi_agent_manager.create_agent(expert)
