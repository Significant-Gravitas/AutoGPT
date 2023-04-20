from colorama import Fore
import json
import importlib

from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger
from autogpt.promptgenerator import PromptGenerator
from autogpt.config import Config
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

CFG = Config()


def log_err(s: str, headline: str = "Error: \n"):
    logger.error(headline, s)


def _extract_yaml_config_type(yaml_config_entry: dict):
    if "raw" in yaml_config_entry:
        return "raw"
    if "collection" in yaml_config_entry:
        return "collection"
    log_err(f"Unknown config type given for entry: {json.dumps(yaml_config_entry)}")


def _load_yaml_config_into_prompt_generator(
    prompt_config,
    prompt_generator,
    yaml_key
):
    for yaml_entry in prompt_config[yaml_key]:
        yaml_entry_type = _extract_yaml_config_type(yaml_entry)

        if yaml_entry_type == "raw":
            prompt_generator.add_resource(yaml_entry["raw"])

        elif yaml_entry_type == "collection":
            yaml_entry_module = importlib.import_module(yaml_entry["collection"])

            if hasattr(yaml_entry_module, yaml_key):
                yaml_entry_collection = getattr(yaml_entry_module, yaml_key)
            else:
                log_err(
                    f"Failed to load module from yaml: {yaml_entry_module}"
                )
                continue

            if not isinstance(yaml_entry_collection, list):
                log_err(
                    f"Invalid {yaml_key} collection given in yaml config (it should be"
                    f" of type list[str]): {yaml_entry['collection']}"
                )
                continue

            for yaml_entry_str in yaml_entry_collection:
                prompt_generator.add_constraint(yaml_entry_str)


def get_prompt() -> str:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the Config object
    cfg = Config()

    # Grab the prompt config directly for ease later
    prompt_config = cfg.yaml_config["prompt"]

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    _load_yaml_config_into_prompt_generator(
        prompt_config,
        prompt_generator,
        "constraints"
    )

    # Define the command list
    commands = [
        ("Google Search", "google", {"input": "<search>"}),
        (
            "Browse Website",
            "browse_website",
            {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
        ),
        (
            "Start GPT Agent",
            "start_agent",
            {"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"},
        ),
        (
            "Message GPT Agent",
            "message_agent",
            {"key": "<key>", "message": "<message>"},
        ),
        ("List GPT Agents", "list_agents", {}),
        ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),
        (
            "Clone Repository",
            "clone_repository",
            {"repository_url": "<url>", "clone_path": "<directory>"},
        ),
        ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
        ("Read file", "read_file", {"file": "<file>"}),
        ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
        ("Delete file", "delete_file", {"file": "<file>"}),
        ("Search Files", "search_files", {"directory": "<directory>"}),
        ("Evaluate Code", "evaluate_code", {"code": "<full_code_string>"}),
        (
            "Get Improved Code",
            "improve_code",
            {"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"},
        ),
        (
            "Write Tests",
            "write_tests",
            {"code": "<full_code_string>", "focus": "<list_of_focus_areas>"},
        ),
        ("Execute Python File", "execute_python_file", {"file": "<file>"}),
        ("Generate Image", "generate_image", {"prompt": "<prompt>"}),
        ("Send Tweet", "send_tweet", {"text": "<text>"}),
    ]

    # Only add the audio to text command if the model is specified
    if cfg.huggingface_audio_to_text_model:
        commands.append(
            (
                "Convert Audio to text",
                "read_audio_from_file",
                {"file": "<file>"}
            ),
        )

    # Only add shell command to the prompt if the AI is allowed to execute it
    if cfg.execute_local_commands:
        commands.append(
            (
                "Execute Shell Command, non-interactive commands only",
                "execute_shell",
                {"command_line": "<command_line>"},
            ),
        )

    # Add these command last.
    commands.append(
        ("Do Nothing", "do_nothing", {}),
    )
    commands.append(
        ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
    )

    # Add commands to the PromptGenerator object
    for command_label, command_name, args in commands:
        prompt_generator.add_command(command_label, command_name, args)

    # Add resources to the PromptGenerator object
    _load_yaml_config_into_prompt_generator(
        prompt_config,
        prompt_generator,
        "resources"
    )

    # Add performance evaluations to the PromptGenerator object
    _load_yaml_config_into_prompt_generator(
        prompt_config,
        prompt_generator,
        "evaluations"
    )

    # Generate the prompt string
    return prompt_generator.generate_prompt_string()


def construct_prompt() -> str:
    """Construct the prompt for the AI to respond to

    Returns:
        str: The prompt string
    """
    config = AIConfig.load(CFG.ai_settings_file)
    if CFG.skip_reprompt and config.ai_name:
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
        should_continue = clean_input(
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
        config.save(CFG.ai_settings_file)

    # Get rid of this global:
    global ai_name
    ai_name = config.ai_name

    return config.construct_full_prompt()
