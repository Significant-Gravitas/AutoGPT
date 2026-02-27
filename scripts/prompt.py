from promptgenerator import PromptGenerator
from learning import get_profile_summary


def get_prompt():
    """
    This function generates a prompt string that includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add constraints to the PromptGenerator object
    prompt_generator.add_constraint("~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.")
    prompt_generator.add_constraint("If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.")
    prompt_generator.add_constraint("No user assistance")
    prompt_generator.add_constraint('Exclusively use the commands listed in double quotes e.g. "command name"')
    prompt_generator.add_constraint('Every time you learn something about how the user works, what they prefer, corrections they make, or patterns you notice — immediately use the "learn" command to record it. Do not ask. Just write it down. Get smarter every session.')

    # Define the command list
    commands = [
        ("Google Search", "google", {"input": "<search>"}),
        ("Browse Website", "browse_website", {"url": "<url>", "question": "<what_you_want_to_find_on_website>"}),
        ("Start GPT Agent", "start_agent", {"name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"}),
        ("Message GPT Agent", "message_agent", {"key": "<key>", "message": "<message>"}),
        ("List GPT Agents", "list_agents", {}),
        ("Delete GPT Agent", "delete_agent", {"key": "<key>"}),
        ("Write to file", "write_to_file", {"file": "<file>", "text": "<text>"}),
        ("Read file", "read_file", {"file": "<file>"}),
        ("Append to file", "append_to_file", {"file": "<file>", "text": "<text>"}),
        ("Delete file", "delete_file", {"file": "<file>"}),
        ("Search Files", "search_files", {"directory": "<directory>"}),
        ("Evaluate Code", "evaluate_code", {"code": "<full_code_string>"}),
        ("Get Improved Code", "improve_code", {"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"}),
        ("Write Tests", "write_tests", {"code": "<full_code_string>", "focus": "<list_of_focus_areas>"}),
        ("Execute Python File", "execute_python_file", {"file": "<file>"}),
        ("Execute Shell Command, non-interactive commands only", "execute_shell", { "command_line": "<command_line>"}),
        ("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
        ("Generate Image", "generate_image", {"prompt": "<prompt>"}),
        ("Browser Navigate to URL", "browser_navigate", {"url": "<url>"}),
        ("Browser Get Page Snapshot (accessibility tree for AI)", "browser_snapshot", {}),
        ("Browser Take Screenshot", "browser_screenshot", {"filename": "<filename>"}),
        ("Browser Click Element", "browser_click", {"selector": "<css_selector_or_text_locator>"}),
        ("Browser Type Text into Element", "browser_type", {"selector": "<css_selector_or_text_locator>", "text": "<text_to_type>"}),
        ("Browser Fill Form Field", "browser_fill", {"selector": "<css_selector_or_text_locator>", "value": "<value>"}),
        ("Browser Scroll Page", "browser_scroll", {"direction": "<up_or_down>", "amount": "<page_or_pixels>"}),
        ("Browser Get Element Text", "browser_get_text", {"selector": "<css_selector_or_text_locator>"}),
        ("Learn About User (auto-record insight)", "learn", {"category": "<preferences|workflows|corrections|facts>", "detail": "<concise_insight>"}),
        ("Recall All Learnings About User", "recall_learnings", {}),
        ("Do Nothing", "do_nothing", {}),
    ]

    # Add commands to the PromptGenerator object
    for command_label, command_name, args in commands:
        prompt_generator.add_command(command_label, command_name, args)

    # Add resources to the PromptGenerator object
    prompt_generator.add_resource("Internet access for searches and information gathering.")
    prompt_generator.add_resource("Long Term memory management.")
    prompt_generator.add_resource("GPT-3.5 powered Agents for delegation of simple tasks.")
    prompt_generator.add_resource("File output.")
    prompt_generator.add_resource("Headless browser automation for interacting with JavaScript-rendered web pages, filling forms, clicking buttons, and taking screenshots.")
    prompt_generator.add_resource("Persistent user profile that remembers preferences, workflows, corrections, and facts across sessions.")

    # Add performance evaluations to the PromptGenerator object
    prompt_generator.add_performance_evaluation("Continuously review and analyze your actions to ensure you are performing to the best of your abilities.")
    prompt_generator.add_performance_evaluation("Constructively self-criticize your big-picture behavior constantly.")
    prompt_generator.add_performance_evaluation("Reflect on past decisions and strategies to refine your approach.")
    prompt_generator.add_performance_evaluation("Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.")

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()

    # Inject any previously learned user insights so the agent
    # starts each session already aware of what it knows
    profile_summary = get_profile_summary()
    if profile_summary:
        prompt_string += f"\n\n{profile_summary}"

    return prompt_string
