"""Main script for the autogpt package."""
import click


@click.group(invoke_without_command=True)
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-settings",
    "-C",
    help="Specifies which ai_settings.yaml file to use, will also automatically skip the re-prompt.",
)
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option("--debug", is_flag=True, help="Enable Debug Mode")
@click.option("--gpt3only", is_flag=True, help="Enable GPT3.5 Only Mode")
@click.option("--gpt4only", is_flag=True, help="Enable GPT4 Only Mode")
@click.option(
    "--use-memory",
    "-m",
    "memory_type",
    type=str,
    help="Defines which Memory backend to use",
)
@click.option(
    "-b",
    "--browser-name",
    help="Specifies which web-browser to use when using selenium to scrape the web.",
)
@click.option(
    "--allow-downloads",
    is_flag=True,
    help="Dangerous: Allows Auto-GPT to download files natively.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.pass_context
def main(
    ctx: click.Context,
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    skip_reprompt: bool,
    speak: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
) -> None:
    """
    Welcome to AutoGPT an experimental open-source application showcasing the capabilities of the GPT-4 pushing the boundaries of AI.

    Start an Auto-GPT assistant.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    import logging
    import sys

    from colorama import Fore

    from autogpt.agent.agent import Agent
    from autogpt.commands.command import CommandRegistry
    from autogpt.config import Config, check_openai_api_key
    from autogpt.configurator import create_config
    from autogpt.logs import logger
    from autogpt.memory import get_memory
    from autogpt.plugins import scan_plugins
    from autogpt.prompts.prompt import construct_main_ai_config
    from autogpt.utils import get_current_git_branch, get_latest_bulletin

    if ctx.invoked_subcommand is None:
        cfg = Config()
        # TODO: fill in llm values here
        check_openai_api_key()
        create_config(
            continuous,
            continuous_limit,
            ai_settings,
            skip_reprompt,
            speak,
            debug,
            gpt3only,
            gpt4only,
            memory_type,
            browser_name,
            allow_downloads,
            skip_news,
        )
        logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
        ai_name = ""
        if not cfg.skip_news:
            motd = get_latest_bulletin()
            if motd:
                logger.typewriter_log("NEWS: ", Fore.GREEN, motd)
            git_branch = get_current_git_branch()
            if git_branch and git_branch != "stable":
                logger.typewriter_log(
                    "WARNING: ",
                    Fore.RED,
                    f"You are running on `{git_branch}` branch "
                    "- this is not a supported branch.",
                )
            if sys.version_info < (3, 10):
                logger.typewriter_log(
                    "WARNING: ",
                    Fore.RED,
                    "You are running on an older version of Python. "
                    "Some people have observed problems with certain "
                    "parts of Auto-GPT with this version. "
                    "Please consider upgrading to Python 3.10 or higher.",
                )

        cfg = Config()
        cfg.set_plugins(scan_plugins(cfg, cfg.debug_mode))
        # Create a CommandRegistry instance and scan default folder
        command_registry = CommandRegistry()
        command_registry.import_commands("autogpt.commands.analyze_code")
        command_registry.import_commands("autogpt.commands.audio_text")
        command_registry.import_commands("autogpt.commands.execute_code")
        command_registry.import_commands("autogpt.commands.file_operations")
        command_registry.import_commands("autogpt.commands.git_operations")
        command_registry.import_commands("autogpt.commands.google_search")
        command_registry.import_commands("autogpt.commands.image_gen")
        command_registry.import_commands("autogpt.commands.improve_code")
        command_registry.import_commands("autogpt.commands.twitter")
        command_registry.import_commands("autogpt.commands.web_selenium")
        command_registry.import_commands("autogpt.commands.write_tests")
        command_registry.import_commands("autogpt.app")
        ai_name = ""
        ai_config = construct_main_ai_config()
        ai_config.command_registry = command_registry
        # print(prompt)
        # Initialize variables
        full_message_history = []
        next_action_count = 0
        # Make a constant:
        triggering_prompt = (
            "Determine which next command to use, and respond using the"
            " format specified above:"
        )
        # Initialize memory and make sure it is empty.
        # this is particularly important for indexing and referencing pinecone memory
        memory = get_memory(cfg, init=True)
        logger.typewriter_log(
            "Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
        )
        logger.typewriter_log("Using Browser:", Fore.GREEN, cfg.selenium_web_browser)
        system_prompt = ai_config.construct_full_prompt()
        if cfg.debug_mode:
            logger.typewriter_log("Prompt:", Fore.GREEN, system_prompt)
        agent = Agent(
            ai_name=ai_name,
            memory=memory,
            full_message_history=full_message_history,
            next_action_count=next_action_count,
            command_registry=command_registry,
            config=ai_config,
            system_prompt=system_prompt,
            triggering_prompt=triggering_prompt,
        )
        agent.start_interaction_loop()


if __name__ == "__main__":
    main()
