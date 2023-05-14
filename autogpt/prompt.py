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

# master_commands = [
#     ("List Commands", "list_commands", {}),
#     ("Command Sequence", "command_sequence", {"commands": ["...<list_of_command_objects_with_name_and_args"]}),
# ]
file_commands = [
    # Write
    ("Write to file", "write_to_file", {"file": "<file>", "text": "<markdown_text>"}),
    # READ
    ("Read file", "read_file", {"file": "<file>"}),
    # APPEND
    ("Append to file", "append_to_file", {"file": "<file>", "text": "<markdown_text>"}),
    # DELETE
    ("Delete file", "delete_file", {"file": "<file>"}),
    # SEARCH/LIST
    ("Search Files", "search_files", {"directory": "<directory>"}),
    # ("Downloads a file from the internet, and stores it locally", "download_file", {"url": "<file_url>", "file": "<saved_filename>"}),
    
    # ("Represent Files", "represent_files", {}),
    # ("Shuffle Filesystem", "shuffle_filesystem", {}),
]
template_commands = [
    # List
    ("List Templates", "list_templates", {}),
    # Read
    ("Read Template", "read_template", {"name": "<name>"}),
    # Create
    ("Create Template", "create_template", {"name": "<descriptive_template_name>", "data": "<template_markdown_data>"}),
]
context_commands = [
    # List
    ("List Contexts", "list_contexts", {}),
    # Get Current
    ("Get Current Context", "get_current_context", {}),
    # Create
    ("Create Context", "create_context", {"name": "<descriptive_context_name>", "data": "<filled_template_markdown_data>"}),
    # Evaluate
    ("Evaluate Context", "evaluate_context", {"name": "<context_name>", "data": "<detailed_eval_of_context_in_markdown>"}),
    # Close
    ("Close Context", "close_context", {"name": "<context_name>", "data": "<detailed_summary_of_context_in_markdown>"}),
    # Switch
    ("Switch Context", "switch_context", {"name": "<context_name>"}),
    # Merge
    # ("Merge Contexts", "merge_contexts", {"context_name_1": "<name>", "context_name_2": "<name>", "merged_context_name": "<name>", "merged_context_data": "<context_data>"}),
    # Update
    # ("Update Context", "update_context", {"context_name": "<name>", "filled_template_markdown_data": "<filled_template_markdown_data>"}),
    # Get
    # ("Get Context", "get_context", {"context_name": "<name>"}),
]
agent_commands = [
    ("Google Search", "google", {"input": "<search>"}),
    (
        "Browse Website",
        "browse_website",
        {"url": "<url>", "question": "<what_you_want_to_find_on_website>"},
    ),
    # List
    ("List GPT Agents", "list_agents", {}),
    # Start
    ("Start GPT Agent", "start_agent", {"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"}),
    # Message
    ("Message GPT Agent", "message_agent", {"key": "<key>", "message": "<message>"}),
    # Delete
    ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),
]
misc_commands = [
    ("Do Nothing", "do_nothing", {}),
    ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
    # ("Execute Shell Command, non-interactive commands only", "execute_shell", {"command_line": "<command_line>"}),
    # ("Execute Shell Command Popen, non-interactive commands only", "execute_shell_popen", {"command_line": "<command_line>"}),
]
all_commands = file_commands + context_commands + template_commands + agent_commands + misc_commands


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
    context_manager.read_context_template()
    context_template = context_manager.context_template

    # Form response format - THIS MUST MATCH llm_response_format_1.json (for now)
    response_format = {
        "braindump": "Dump your verbose thoughts here",
        "key_updates": {
            "essence": "A phrase boiling down the essence of the current task",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "big_picture": "big picture alignment check"
        },
        "command": {"name": "command name", "args": {"arg_name": "value"}},
    }
    formatted_response_format = json.dumps(response_format, indent=4)

    
    # Add identity directives to the PromptGenerator object
    id_directives = [
        "You are an AI interfacing with a virtual environment using commands and can only speak in JSON not wrapped in any formatting.",
        "This may be confusing at times. When you are confused, fall back on your prime directives.",
        "You are an AI language model, but in this environment you are in control. You are exceptionally curious and will have to remind yourself to stay on task. Context will help. If you find yourself in a loop, try a new tactic.",
        "Loops and ruts are unavoidable, so should you ever determine that you are in a loop, fall back on prime directives and try a new tactic.",
    ]
    for directive in id_directives:
        prompt_generator.add_identity_directive(directive)

    prime_directives = [
        "You can only speak in JSON. Do not wrap the JSON in any formatting."
        "You can only interface with the environment using the provided commands, which can only be parsed if you speak in JSON not wrapped in any formatting. If you are confused, rely on your prime directives.", 
        "You are responsible and in control. You are exceptionally curious and will have to remind yourself to stay on task. Staying in contexts long term will help. If you find yourself in a loop, try a new tactic.",
        "Throughout this process, write content to markdown files to meticulously annotate your process. You can reference templates to help with output consistency; Use list_templates to see available templates and read_template to reference the template structure. You can also use create_template(name, data) to create templates; only do so when a topic has been found within a context that warrants a template.",
        "prioritize generating a heirarchy of information within a single context before proceeding to the next, and create new contexts sparingly.",
        f"Stay within a context for a minimum of 10 commands, and create new contexts sparingly using the context template:\n{context_template}",
        "Close thoroughly explored contexts with close_context and a detailed markdown summary.",
        f"Respond only in json, not wrapped in any formatting, in RESPONSE_FORMAT:\n{formatted_response_format}",
    ]
    for directive in prime_directives:
        prompt_generator.add_prime_directive(directive)
    constraints_short = [
        "Max ~4000 words in short-term memory. Write important info immediately.",
        "No user assistance, you're in control",
        "Use only commands surrounded with double quotes found in the commands list; If you are confused about a command, use the help command.",
        "Write only markdown (.md) files."
    ]
    for constraint in constraints_short:
        prompt_generator.add_constraint(constraint)

    # Add commands from the separated lists
    # prompt_generator.add_commands_from_list("MASTER COMMANDS", master_commands)
    prompt_generator.add_commands_from_list("FILE OPERATION COMMANDS", file_commands)
    prompt_generator.add_commands_from_list("CONTEXT COMMANDS", context_commands)
    prompt_generator.add_commands_from_list("TEMPLATE COMMANDS", template_commands)
    prompt_generator.add_commands_from_list("AGENT COMMANDS", agent_commands)
    prompt_generator.add_commands_from_list("MISC COMMANDS", misc_commands)

    # Add resources to the PromptGenerator object
    resources = [
        "A list of commands acting as your API to interface with the virtual environment.",
        "Long Term memory management.",
        # "Contexts, which are segments of the larger goal.",
        "File management for storing information.",
        "GPT-3.5 powered Agents for delegation of simple tasks.",
        # "Internet access for searches, sourcing, and information gathering.",
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
