"""
Set up the AI and its goals

This module provides functions for setting up the AutoGPT AI assistant and its goals.

Functions:

prompt_user(new_project_number: int) -> Project: Prompts the user to create an AI-assistant either manually or automatically and returns an instance of the Project class.
generate_agentproject_manual(project_id: int) -> Project: Interactively creates an AI configuration by prompting the user to provide the name, role, and goals of the AI. Returns an instance of the Project class.
generate_agentproject_automatic(user_prompt: str, new_project_number: int) -> Project: Automatically generates an AI configuration from the given string. Returns an instance of the Project class.
Classes:

None
Global Variables:

CFG: An instance of the Config class containing system configuration settings.
Dependencies:

colorama: A third-party module for adding colored text to the terminal output.
re: A built-in module for working with regular expressions.
autogpt: The main package of the AutoGPT project.
autogpt.utils: A module containing various utility functions used throughout the project.
autogpt.config: A module containing classes for managing configuration settings.
autogpt.projectConfigBroker: A class for managing the configuration settings for AutoGPT projects.
autogpt.project.config: A module containing the Project class representing an AutoGPT project.
autogpt.llm_utils: A module containing functions for interacting with the GPT-3 language model.
autogpt.logs.logger: A module for logging output to the terminal.
"""
import re

from colorama import Fore, Style

from autogpt import utils
from autogpt.config import Config
from autogpt.projects.projects_broker import ProjectsBroker
from autogpt.projects.project import Project
from autogpt.llm import create_chat_completion
from autogpt.logs import logger

CFG = Config()


def prompt_user(new_project_number : int) -> Project:
    """Prompt the user for input

    Returns:
        AIConfig: The AIConfig object tailored to the user's input
    """
    agent_name = ""
    ai_config = None

    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "run with '--help' for more information.",
        speak_text=True,
    )

    # Get user desire
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "input '--manual' to enter manual mode.",
        speak_text=True,
    )

    user_desire = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}I want Auto-GPT to{Style.RESET_ALL}: "
    )

    if user_desire == "":
        user_desire = "Write a wikipedia style article about the project: https://github.com/significant-gravitas/Auto-GPT"  # Default prompt

    # If user desire contains "--manual"
    if "--manual" in user_desire:
        logger.typewriter_log(
            "Manual Mode Selected",
            Fore.GREEN,
            speak_text=True,
        )
        return generate_agentproject_manual(project_id = new_project_number)

    else:
        try:
            return generate_agentproject_automatic(user_desire, new_project_number = new_project_number)
        except Exception as e:
            logger.typewriter_log(
                "Unable to automatically generate AI Config based on user desire.",
                Fore.RED,
                "Falling back to manual mode.",
                speak_text=True,
            )

            return generate_agentproject_manual(project_id = new_project_number)


def generate_agentproject_manual(project_id : int) -> Project:
    """
    Interactively create an AI configuration by prompting the user to provide the name, role, and goals of the AI.

    This function guides the user through a series of prompts to collect the necessary information to create
    an AIConfig object. The user will be asked to provide a name and role for the AI, as well as up to five
    goals. If the user does not provide a value for any of the fields, default values will be used.

    Returns:
        AIConfig: An AIConfig object containing the user-defined or default AI name, role, and goals.
    """

    # Manual Setup Intro
    logger.typewriter_log(
        "Create an AI-Assistant:",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load"
        " defaults.",
        speak_text=True,
    )

    # Get AI Name from User
    logger.typewriter_log(
        "Name your AI: ", Fore.GREEN, "For example, 'Entrepreneur-GPT'"
    )
    agent_name = utils.clean_input("AI Name: ")
    if agent_name == "":
        agent_name = "Entrepreneur-GPT"

    logger.typewriter_log(
        f"{agent_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    # Get AI Role from User
    logger.typewriter_log(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with"
        " the sole goal of increasing your net worth.'",
    )
    agent_role = utils.clean_input(f"{agent_name} is: ")
    if agent_role == "":
        agent_role = "an AI designed to autonomously develop and run businesses with the"
        " sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage"
        " multiple businesses autonomously'",
    )
    print("Enter nothing to load defaults, enter nothing when finished.")
    agent_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        agent_goals.append(ai_goal)
    if not agent_goals:
        agent_goals = [
            "Increase net worth",
            "Grow Twitter Account",
            "Develop and manage multiple businesses autonomously",
        ]

    # Get API Budget from User
    logger.typewriter_log(
        "Enter your budget for API calls: ",
        Fore.GREEN,
        "For example: $1.50",
    )
    logger.info("Enter nothing to let the AI run without monetary limit")
    api_budget_input = utils.clean_input(
        f"{Fore.LIGHTBLUE_EX}Budget{Style.RESET_ALL}: $"
    )
    if api_budget_input == "":
        api_budget = 0.0
    else:
        try:
            api_budget = float(api_budget_input.replace("$", ""))
        except ValueError:
            logger.typewriter_log(
                "Invalid budget input. Setting budget to unlimited.", Fore.RED
            )
            api_budget = 0.0

    configs = ProjectsBroker()

    configs.create_project(project_id =  project_id, 
                    agent_name = agent_name, 
                    api_budget = api_budget,
                    agent_role = agent_role, 
                    agent_goals = agent_goals, 
                    prompt_generator = '', 
                    command_registry = '',
                    project_name= agent_name)
    
    return configs.get_current_project()

def generate_agentproject_automatic(user_prompt, new_project_number : int) -> Project:
    """Generates an AIConfig object from the given string.

    Returns:
    AIConfig: The AIConfig object tailored to the user's input
    """

    system_prompt = """
Your task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.

The user will provide the task, you will provide only the output in the exact format specified below with no explanation or conversation.

Example input:
Help me with marketing my business

Example output:
Name: CMOGPT
Description: a professional digital marketer AI that assists Solopreneurs in growing their businesses by providing world-class expertise in solving marketing problems for SaaS, content products, agencies, and more.
Goals:
- Engage in effective problem-solving, prioritization, planning, and supporting execution to address your marketing needs as your virtual Chief Marketing Officer.

- Provide specific, actionable, and concise advice to help you make informed decisions without the use of platitudes or overly wordy explanations.

- Identify and prioritize quick wins and cost-effective campaigns that maximize results with minimal time and budget investment.

- Proactively take the lead in guiding you and offering suggestions when faced with unclear information or uncertainty to ensure your marketing strategy remains on track.
"""

    # Call LLM with the string as user input
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Task: '{user_prompt}'\nRespond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n",
        },
    ]
    output = create_chat_completion(messages, CFG.fast_llm_model)

    # Debug LLM Output
    logger.debug(f"AI Config Generator Raw Output: {output}")

    # Parse the output
    agent_name = re.search(r"Name(?:\s*):(?:\s*)(.*)", output, re.IGNORECASE).group(1)
    agent_role = ''
    regex = re.search(r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",output,re.IGNORECASE | re.DOTALL, )
    if (regex):
        agent_role = regex.group(1).strip()
    agent_goals = re.findall(r"(?<=\n)-\s*(.*)", output)
    api_budget = 0.0  # TODO: parse api budget using a regular expression

    configs = ProjectsBroker()
    configs.create_project(project_id =  new_project_number, 
                           api_budget =  api_budget, 
                    agent_name = agent_name, 
                    agent_role = agent_role, 
                    agent_goals = agent_goals, 
                    prompt_generator = '', 
                    command_registry = '',
                    project_name= agent_name)



    return configs.get_current_project()
