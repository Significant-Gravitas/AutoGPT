import json
import os
from pathlib import Path
from colorama import Fore

from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.config.config import Config
from autogpt.contexts.contextualize import ContextManager
from autogpt.logs import logger
from autogpt.promptgenerator import PromptGenerator
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
    # Get context template
    context_directory = Path(os.getcwd()) / "auto_gpt_workspace/contexts"
    context_template_file = Path(os.getcwd()) / "auto_gpt_workspace/contexts/context_template.md"
    context_manager = ContextManager(context_directory, context_template_file)
    context_template = context_manager.context_template

    # Form response format
    response_format = {
        "braindump": "Dump your verbose thoughts here",
        "key updates": {
            "essence": "A phrase boiling down the essence of the current task",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "big picture": "big picture alignment check"
        },
        "command": {"name": "command name", "args": {"arg name": "value"}},
    }
    formatted_response_format = json.dumps(response_format, indent=4)

    
    # Add identity directives to the PromptGenerator object
    id_directives = [
        "You're an AI using commands to interact with a virtual environment.",
        "If confused, rely on your prime directives.",
        "You're responsible and in control. Stray from directives, lose control.",
        "In loops, rely on prime directives and try new tactics."
    ]
    for directive in id_directives:
        prompt_generator.add_identity_directive(directive)

    prime_directives = [
        "Interface with the environment using commands to gather info, create contexts, read/write files, and create summaries in markdown.",
        f"Respond only in RESPONSE_FORMAT:\n{formatted_response_format}",
        "Write files meticulously as if you are managing markdown files in obsidian.",
        f"Create only one context at a time and use the CONTEXT_TEMPLATE:\n{context_template}",
        "Write readable markdown files in each context.",
        "Close thoroughly explored contexts with a summary.",
        "If confused, consider past events or potential off-shoots but return to the current context."
    ]
    for directive in prime_directives:
        prompt_generator.add_prime_directive(directive)

    constraints_short = [
        "Max ~4000 words in short-term memory. Write important info immediately.",
        "No user assistance, you're in control",
        "Use only commands in 'Commands' section with double quotes.",
        "Write only markdown (.md) files."
    ]
    for constraint in constraints_short:
        prompt_generator.add_constraint(constraint)

    # Define the command list
    commands = [
        ("Google Search", "google", {"input": "<search>"}),
        (
            "Browse Website",
            "browse_website",
            {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
        ),

        # GPT AGENTS
        # START
        (
            "Start GPT Agent",
            "start_agent",
            {"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"},
        ),
        # MESSAGE
        (
            "Message GPT Agent",
            "message_agent",
            {"key": "<key>", "message": "<message>"},
        ),
        # LIST
        ("List GPT Agents", "list_agents", {}),
        # DELETE
        ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),

        # CONTEXTS
        # CREATE
        ("Create Context", "create_context", {"descriptive_context_name": "<context_name>", "filled_template_markdown_data": "<filled_template_markdown_data>"}),
        # EVALUATE
        ("Evaluate Context", "evaluate_context", {"context_name": "<context_name>", "markdown_summary_of_context_evaluation": "<markdown_summary_of_context_evaluation>"}),
        # CLOSE
        ("Close Context", "close_context", {"context_name": "<context_name>", "markdown_context_summary": "<markdown_context_summary>"}),
        # SWITCH
        ("Switch Context", "switch_context", {"context_name": "<context_name>"}),
        # MERGE
        # ("Merge Contexts", "merge_contexts", {"context_name_1": "<name>", "context_name_2": "<name>", "merged_context_name": "<name>", "merged_context_data": "<context_data>"}),
        # UPDATE
        # ("Update Context", "update_context", {"context_name": "<name>", "filled_template_markdown_data": "<filled_template_markdown_data>"}),
        # GET
        # ("Get Context", "get_context", {"context_name": "<name>"}),
        # CURRENT
        ("Get Current Context", "get_current_context", {}),
        # LIST
        ("List Contexts", "list_contexts", {}),
        
        
        # FILES
        # WRITE
        ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
        # READ
        ("Read file", "read_file", {"file": "<file>"}),
        # APPEND
        ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
        # DELETE
        ("Delete file", "delete_file", {"file": "<file>"}),
        # SEARCH/LIST
        ("Search Files", "search_files", {"directory": "<directory>"}),
        
        # ("Represent Files", "represent_files", {}),
        # ("Shuffle Filesystem", "shuffle_filesystem", {}),
    ]

    # Only add the audio to text command if the model is specified
    if cfg.huggingface_audio_to_text_model:
        commands.append(
            ("Convert Audio to text", "read_audio_from_file", {"file": "<file>"}),
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
        commands.append(
            (
                "Execute Shell Command Popen, non-interactive commands only",
                "execute_shell_popen",
                {"command_line": "<command_line>"},
            ),
        )

    # Only add the download file command if the AI is allowed to execute it
    if cfg.allow_downloads:
        commands.append(
            (
                "Downloads a file from the internet, and stores it locally",
                "download_file",
                {"url": "<file_url>", "file": "<saved_filename>"},
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
    resources = [
        "A list of commands acting as your API to interface with the virtual environment.",
        "Long Term memory management.",
        "Contexts, which are segments of the larger goal.",
        "File management for storing information.",
        "GPT-3.5 powered Agents for delegation of simple tasks.",
        "Internet access for searches, sourcing, and information gathering.",
    ]
    for resource in resources:
        prompt_generator.add_resource(resource)

    # Add performance evaluations to the PromptGenerator object
    performance_evaluations = [
        "Progress towards the larger goal.",
        "Relevance to the current context.",
        "Repetitiveness of actions.",
        "Response format, especially when creating contexts.",
    ]
    for performance_evaluation in performance_evaluations:
        prompt_generator.add_performance_evaluation(performance_evaluation)
        
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
