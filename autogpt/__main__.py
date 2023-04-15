"""Main script for the autogpt package."""
import logging
from colorama import Fore
from autogpt.agent.agent import Agent
from autogpt.args import parse_arguments

from autogpt.config import Config, check_openai_api_key
from autogpt.logs import logger
from autogpt.memory import get_memory

from autogpt import chat
from autogpt import commands as cmd
from autogpt import speak, utils
from autogpt.agent import Agent
from autogpt.ai_config import AIConfig
from autogpt.config import Config
from autogpt.json_parser import fix_and_parse_json
from autogpt.logger import logger
from autogpt.memory import get_memory, get_supported_memory_backends
from autogpt.spinner import Spinner

# Load environment variables from .env file


def check_openai_api_key():
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not cfg.openai_api_key:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
        )
        print("You can get your key from https://beta.openai.com/account/api-keys")
        exit(1)


def construct_prompt():
    """Construct the prompt for the AI to respond to"""
    config: AIConfig = AIConfig.load(cfg.ai_settings_file)
    if cfg.skip_reprompt and config.ai_name:
        logger.typewriter_log("Name :", Fore.GREEN, config.ai_name)
        logger.typewriter_log("Role :", Fore.GREEN, config.ai_role)
        logger.typewriter_log("Goals:", Fore.GREEN, f"{config.ai_goals}")
    elif config.ai_name:
        logger.typewriter_log(
            "Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config.ai_name}?",
            speak_text=True,
        )
        should_continue = utils.clean_input(
            f"""Continue with the last settings?
Name:  {config.ai_name}
Role:  {config.ai_role}
Goals: {config.ai_goals}
Continue (y/n): """
        )
        if should_continue.lower() == "n":
            config = AIConfig()

    if not config.ai_name:
        config = prompt_user()
        config.save()

    # Get rid of this global:
    global ai_name
    ai_name = config.ai_name

    return config.construct_full_prompt()


def prompt_user():
    """Prompt the user for input"""
    ai_name = ""
    # Construct the prompt
    logger.typewriter_log(
        "Welcome to Auto-GPT! ",
        Fore.GREEN,
        "Enter the name of your AI and its role below. Entering nothing will load"
        " defaults.",
        speak_text=True,
    )

    # Get AI Name from User
    logger.typewriter_log(
        "Name your AI: ", Fore.GREEN, "For example, 'Entrepreneur-GPT'"
    )
    ai_name = utils.clean_input("AI Name: ")
    if ai_name == "":
        ai_name = "Entrepreneur-GPT"

    logger.typewriter_log(
        f"{ai_name} here!", Fore.LIGHTBLUE_EX, "I am at your service.", speak_text=True
    )

    # Get AI Role from User
    logger.typewriter_log(
        "Describe your AI's role: ",
        Fore.GREEN,
        "For example, 'an AI designed to autonomously develop and run businesses with"
        " the sole goal of increasing your net worth.'",
    )
    ai_role = utils.clean_input(f"{ai_name} is: ")
    if ai_role == "":
        ai_role = "an AI designed to autonomously develop and run businesses with the"
        " sole goal of increasing your net worth."

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "Enter up to 5 goals for your AI: ",
        Fore.GREEN,
        "For example: \nIncrease net worth, Grow Twitter Account, Develop and manage"
        " multiple businesses autonomously'",
    )
    print("Enter nothing to load defaults, enter nothing when finished.", flush=True)
    ai_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = [
            "Increase net worth",
            "Grow Twitter Account",
            "Develop and manage multiple businesses autonomously",
        ]

    config = AIConfig(ai_name, ai_role, ai_goals)
    return config


def parse_arguments():
    """Parses the arguments passed to the script"""
    global cfg
    cfg.set_debug_mode(False)
    cfg.set_continuous_mode(False)
    cfg.set_speak_mode(False)

    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument(
        "--continuous", "-c", action="store_true", help="Enable Continuous Mode"
    )
    parser.add_argument(
        "--continuous-limit",
        "-l",
        type=int,
        dest="continuous_limit",
        help="Defines the number of times to run in continuous mode",
    )
    parser.add_argument("--speak", action="store_true", help="Enable Speak Mode")
    parser.add_argument("--debug", action="store_true", help="Enable Debug Mode")
    parser.add_argument(
        "--gpt3only", action="store_true", help="Enable GPT3.5 Only Mode"
    )
    parser.add_argument("--gpt4only", action="store_true", help="Enable GPT4 Only Mode")
    parser.add_argument(
        "--use-memory",
        "-m",
        dest="memory_type",
        help="Defines which Memory backend to use",
    )
    parser.add_argument(
        "--skip-reprompt",
        "-y",
        dest="skip_reprompt",
        action="store_true",
        help="Skips the re-prompting messages at the beginning of the script",
    )
    parser.add_argument(
        "--ai-settings",
        "-C",
        dest="ai_settings_file",
        help="Specifies which ai_settings.yaml file to use, will also automatically"
        " skip the re-prompt.",
    )
    args = parser.parse_args()

    if args.debug:
        logger.typewriter_log("Debug Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_debug_mode(True)

    if args.continuous:
        logger.typewriter_log("Continuous Mode: ", Fore.RED, "ENABLED")
        logger.typewriter_log(
            "WARNING: ",
            Fore.RED,
            "Continuous mode is not recommended. It is potentially dangerous and may"
            " cause your AI to run forever or carry out actions you would not usually"
            " authorise. Use at your own risk.",
        )
        cfg.set_continuous_mode(True)

        if args.continuous_limit:
            logger.typewriter_log(
                "Continuous Limit: ", Fore.GREEN, f"{args.continuous_limit}"
            )
            cfg.set_continuous_limit(args.continuous_limit)

    # Check if continuous limit is used without continuous mode
    if args.continuous_limit and not args.continuous:
        parser.error("--continuous-limit can only be used with --continuous")

    if args.speak:
        logger.typewriter_log("Speak Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_speak_mode(True)

    if args.gpt3only:
        logger.typewriter_log("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_smart_llm_model(cfg.fast_llm_model)

    if args.gpt4only:
        logger.typewriter_log("GPT4 Only Mode: ", Fore.GREEN, "ENABLED")
        cfg.set_fast_llm_model(cfg.smart_llm_model)

    if args.memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = args.memory_type
        if not chosen in supported_memory:
            logger.typewriter_log(
                "ONLY THE FOLLOWING MEMORY BACKENDS ARE SUPPORTED: ",
                Fore.RED,
                f"{supported_memory}",
            )
            logger.typewriter_log("Defaulting to: ", Fore.YELLOW, cfg.memory_backend)
        else:
            cfg.memory_backend = chosen

    if args.skip_reprompt:
        logger.typewriter_log("Skip Re-prompt: ", Fore.GREEN, "ENABLED")
        cfg.skip_reprompt = True

    if args.ai_settings_file:
        file = args.ai_settings_file

        # Validate file
        (validated, message) = utils.validate_yaml_file(file)
        if not validated:
            logger.typewriter_log("FAILED FILE VALIDATION", Fore.RED, message)
            logger.double_check()
            exit(1)

        logger.typewriter_log("Using AI Settings File:", Fore.GREEN, file)
        cfg.ai_settings_file = file
        cfg.skip_reprompt = True


def main():
    global ai_name, memory
    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    ai_name = ""
    prompt = construct_prompt()
    # print(prompt)
    # Initialize variables
    full_message_history = []
    next_action_count = 0
    # Make a constant:
    user_input = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )
    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    logger.typewriter_log(f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}")
    logger.typewriter_log(f"Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        prompt=prompt,
        user_input=user_input,
    )
    agent.start_interaction_loop()


if __name__ == "__main__":
    main()
