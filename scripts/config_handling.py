import yaml
import data

from colorama import Fore
from ai_config import AIConfig
from console_interaction import print_to_console, prompt_user

def load_variables(config_file="config.yaml"):
    # Load variables from yaml file if it exists
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ai_name = config.get("ai_name")
        ai_role = config.get("ai_role")
        ai_goals = config.get("ai_goals")
    except FileNotFoundError:
        ai_name = ""
        ai_role = ""
        ai_goals = []

    # Prompt the user for input if config file is missing or empty values
    if not ai_name:
        ai_name = input("Name your AI: ")
        if ai_name == "":
            ai_name = "Entrepreneur-GPT"

    if not ai_role:        
        ai_role = input(f"{ai_name} is: ")
        if ai_role == "":
            ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    if not ai_goals:
        print("Enter up to 5 goals for your AI: ")
        print("For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
        print("Enter nothing to load defaults, enter nothing when finished.")
        ai_goals = []
        for i in range(5):
            ai_goal = input(f"Goal {i+1}: ")
            if ai_goal == "":
                break
            ai_goals.append(ai_goal)
        if len(ai_goals) == 0:
            ai_goals = ["Increase net worth", "Grow Twitter Account", "Develop and manage multiple businesses autonomously"]
         
    # Save variables to yaml file
    config = {"ai_name": ai_name, "ai_role": ai_role, "ai_goals": ai_goals}
    with open(config_file, "w") as file:
        documents = yaml.dump(config, file)

    prompt = data.load_prompt()
    prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications."""

    # Construct full prompt
    full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_goals):
        full_prompt += f"{i+1}. {goal}\n"

    full_prompt += f"\n\n{prompt}"
    return full_prompt


def construct_prompt():
    config = AIConfig.load()
    if config.ai_name:
        print_to_console(
            f"Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config.ai_name}?",
            speak_text=True)
        should_continue = input(f"""Continue with the last settings? 
Name:  {config.ai_name}
Role:  {config.ai_role}
Goals: {config.ai_goals}  
Continue (y/n): """)
        if should_continue.lower() == "n":
            config = AIConfig()

    if not config.ai_name:         
        config = prompt_user()
        config.save()

    # Get rid of this global:
    global ai_name
    ai_name = config.ai_name
    
    full_prompt = config.construct_full_prompt()
    return full_prompt