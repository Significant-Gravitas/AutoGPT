import re
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


# expert_prompt = """The task is: {task}.

# Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts. Follow the format precisely: 1. NAME – DESCRIPTION OF HOW THEY ARE THEY ARE USEFUL"""

expert_prompt = """The task is: {task}.

Help me determine which historical or renowned experts in various fields would be best suited to complete a given task, taking into account their specific expertise and access to the internet. Name between {min_experts} and {max_experts} experts and list three goals for them to help the overall task. Follow the format precisely: 
1. NAME: Description of how they are useful
1a) GOAL1
1b) GOAL2
1c) GOAL3"""

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
#     return experts

def parse_experts(experts: str) -> List[Expert]:
    # personas = experts.split(r"[0-9]\. ")
    personas = re.split(r"[0-9]\. ", experts)[1:]
    print(personas)
    res = []
    for persona in personas:
        # tmp = persona.split(r"[0-9][a-c]\) ")
        try: 
            tmp = re.split(r"[0-9][a-c]\) ", persona)
            name, description = tmp[1].split(":")[:2]
            goals = tmp[2:]
            print(name, description, goals)
            res.append(Expert(name, description, goals))
        except:
            print("Error parsing expert")
    return res
    

if __name__ == "__main__":
    # Ask user for task description
    # generate expert list from tasks
    # OPTIONAL: budget experts
    # generate ai_settings.yaml for each expert -- engineering the prompt
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
        experts_string = create_chat_completion([{"role": "system", "content": expert_prompt.format(task=task, min_experts=MIN_EXPERTS, max_experts=MAX_EXPERTS)}], CFG.smart_llm_model, max_tokens=500)
    
    experts = parse_experts(experts_string)
    # TODO: spawn expertGPTs