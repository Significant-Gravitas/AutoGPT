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

from autogpt.config import Config, check_openai_api_key
from autogpt.llm_utils import create_chat_completion
from autogpt.logs import logger
from autogpt.memory import get_memory

from autogpt.prompt import construct_prompt
from autogpt.spinner import Spinner
from multigpt.expert import Expert
from multigpt.multi_agent_manager import MultiAgentManager

MIN_EXPERTS = 1
MAX_EXPERTS = 2
CONTINUOUS_LIMIT = 10
EXPERT_PROMPT = """The task is: {task}.

Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts and list three goals for them to help the overall task. Follow the format precisely: 
1. [Name of the person]: [Description of how they are useful]
1a) [Goal a]
1b) [Goal b]
1c) [Goal c]"""

CFG = Config()
MULTIAGENTMANAGER = MultiAgentManager(CFG)


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
    return res


def create_expert_gpts(experts: List[Expert]):
    # create expertGPTs
    for expert in experts:
        slugified_filename = slugify(expert.ai_name, separator="_", lowercase=True) + "_settings.yaml"
        filepath = os.path.join(os.path.dirname(__file__), "saved_agents", f"{slugified_filename}")
        expert.save(filepath)
        # TODO: sometimes doesn't find the file, because it hasn't finished saving yet
        MULTIAGENTMANAGER.create_agent(expert)



def main() -> None:
    # DONE: Ask user for task description
    # DONE: generate expert list from tasks
    # OPTIONAL: budget experts
    # DONE: generate ai_settings.yaml for each expert -- engineering the prompt
    # create autoGPTs
    # access results of instances -- message passing
    # start_autogpt()

    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if CFG.debug_mode else logging.INFO)
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

    messages = [{"role": "system", "content": EXPERT_PROMPT.format(task=task, min_experts=MIN_EXPERTS,
                                                                   max_experts=MAX_EXPERTS)}]
    with Spinner("Thinking... "):
        experts_string = create_chat_completion(messages=messages, model=CFG.smart_llm_model, max_tokens=1000)

    experts = parse_experts(experts_string)

    logger.typewriter_log(f"Using Browser:", Fore.GREEN, CFG.selenium_web_browser)

    create_expert_gpts(experts)
    MULTIAGENTMANAGER.start_interaction_loop()


if __name__ == "__main__":
    main()
