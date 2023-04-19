import os
from pathlib import Path
from colorama import Fore
from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.contexts.contextualize import ContextManager
from autogpt.logs import logger
from autogpt.promptgenerator import PromptGenerator
from autogpt.config import Config
from autogpt.setup import prompt_user
from autogpt.utils import clean_input

CFG = Config()


def get_prompt() -> str:
    """
    This function generates a prompt string that includes various constraints,
        commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the Config object
    cfg = Config()

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint(
        "~4000 word limit for short term memory. Short term memory is short, so"
        " immediately write important information."
    )
    prompt_generator.add_constraint(
        "When unsure of past events, think about similar events."
    )
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint(
        'Exclusively use the commands listed in double quotes e.g. "command name"'
    )
    prompt_generator.add_constraint(
        "Contexts are a segment of the larger goal. Only create a context when there is enough data to fill the template."
    )

    context_directory = Path(os.getcwd()) / "auto_gpt_workspace/contexts"
    context_template_file = Path(os.getcwd()) / "auto_gpt_workspace/contexts/context_template.md"
    context_manager = ContextManager(context_directory, context_template_file)
    context_template = context_manager.context_template
    prompt_generator.add_constraint(
        "When creating a context, you must use the Context template written in Markdown with no structural changes:\n"
        f"{context_template}\n"
        "Fill in the template with relevant data and use the command \"create_context\" to create the context, passing the markdown text as a string."
    )
    prompt_generator.add_constraint(
        "No code. No Command Line. Exclusively Text. Only write markdown (.md) files."
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
        ("Create Context", "create_context", {"context_name": "<name>", "context_data": "<context_data>"}),
        ("List GPT Agents", "list_agents", {}),
        ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),
        ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
        ("Read file", "read_file", {"file": "<file>"}),
        ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
        ("Delete file", "delete_file", {"file": "<file>"}),
        ("Search Files", "search_files", {"directory": "<directory>"}),
        # ("Represent Files", "represent_files", {}),
        # ("Shuffle Filesystem", "shuffle_filesystem", {}),
    ]

    # Only add the audio to text command if the model is specified
    # if cfg.huggingface_audio_to_text_model:
    #     commands.append(
    #         (
    #             "Convert Audio to text",
    #             "read_audio_from_file",
    #             {"file": "<file>"}
    #         ),
    #     )

    # # Only add shell command to the prompt if the AI is allowed to execute it
    # if cfg.execute_local_commands:
    #     commands.append(
    #         (
    #             "Execute Shell Command, non-interactive commands only",
    #             "execute_shell",
    #             {"command_line": "<command_line>"},
    #         ),
    #     )

    # Add these command last.
    commands.append(
        ("Evaluate Context", "evaluate_context", {}),
    )
    commands.append(
        ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
    )

    # Add commands to the PromptGenerator object
    for command_label, command_name, args in commands:
        prompt_generator.add_command(command_label, command_name, args)

    # Add resources to the PromptGenerator object
    # prompt_generator.add_resource("Reading, writing, appending to, and reorganizing files, which is your primary goal.")
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource("Context creation for segmenting the goal.")
    prompt_generator.add_resource(
        "GPT-3.5 powered Agents for delegation of simple tasks."
    )
    prompt_generator.add_resource(
        "Internet access for searches, sourcing, and information gathering."
    )

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation(
        "Evaluate context usage regularly."
    )
    prompt_generator.add_performance_evaluation(
        "Constructively self-criticize your big-picture behavior and reflect on past decisions and stretegies constantly."
    )
    prompt_generator.add_performance_evaluation(
        "Every command has a cost, so be smart and efficient."
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
