import re
import os
import subprocess
from colorama import Fore, Style
from typing import List

from autogpt.logs import logger
from autogpt import utils
from autogpt.spinner import Spinner
from autogpt.llm_utils import create_chat_completion
from autogpt.config.config import Config

from autogpt.expert import Expert
from autogpt.prompt_engineer import generate_prompt


CFG = Config()
MIN_EXPERTS = 2
MAX_EXPERTS = 6
CONTINUOUS_LIMIT = 10

# expert_prompt = """The task is: {task}.

# Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts. Follow the format precisely: 1. NAME – DESCRIPTION OF HOW THEY ARE THEY ARE USEFUL"""

expert_prompt = """The task is: {task}.

Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts and list three goals for them to help the overall task. Follow the format precisely: 
1. [Name of the person]: [Description of how they are useful]
1a) [Goal a]
1b) [Goal b]
1c) [Goal c]"""

def start_autogpt():
    # check if start programmatically or execute from command line
    pass

# def parse_expert_list(expert_list):
#     # parse expert list
#     # expert list is a string of the form: 1. NAME – DESCRIPTION OF HOW THEY ARE THEY ARE USEFUL
#     # return list of experts
#     experts = []
#     for i in range (1, MAX_EXPERTS + 1):
#         name, description = expert_list.split(f"{i}. ")[1].split("–")[:2]
#         description = description.split(f"{i+1}")[0].strip()
#         experts.append(Expert(name, description, []))
#         # TODO: detect if the model refused to answer
#         # print(f"{experts[-1].name}: {experts[-1].role}")
#     return experts'

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
                goals_str += f"{i+1}. {goal}\n"
            logger.typewriter_log(
                f"Goals:", Fore.GREEN, goals_str
            )
            # print(name, description, goals)
            res.append(Expert(name, description, goals))
        except :
            print("Error parsing expert")
    return res


def spawn_expert_gpt(expert_config_path: str):
    # spawn expertGPT
    # TODO: returns file not found error
    subprocess.run(f"python -m {os.path.join(os.path.dirname(__file__), 'autogpt')} -c --continuous-limit {CONTINUOUS_LIMIT} --ai-settings '{expert_config_path}'")


def create_expert_gpts(experts: List[Expert]):
    # create expertGPTs
    for expert in experts:
        filepath = os.path.join(os.path.dirname(__file__), "multigpt", f"{expert.ai_name}_settings.yaml")
        expert.save(filepath)
        # TODO: sometimes doesn't find the file, because it hasn't finished saving yet
        spawn_expert_gpt(filepath)
        

if __name__ == "__main__":
    # DONE: Ask user for task description
    # DONE: generate expert list from tasks
    # OPTIONAL: budget experts
    # DONE: generate ai_settings.yaml for each expert -- engineering the prompt
    # create autoGPTs
    # access results of instances -- message passing
    # start_autogpt()
    
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

    with Spinner("Thinking... "):
        experts_string = create_chat_completion([{"role": "system", "content": expert_prompt.format(task=task, min_experts=MIN_EXPERTS, max_experts=MAX_EXPERTS)}], CFG.smart_llm_model, max_tokens=1000)
    
    experts = parse_experts(experts_string)
    create_expert_gpts(experts)