""""
This script provides the entry point for the AutoGPT package. It uses the Click library to create a command-line interface for the AutoGPT system.

Description:

This script provides a command-line interface to start the AutoGPT assistant with various options.
The user can enable/disable Continuous Mode, Speak Mode, Debug Mode, GPT3.5 Only Mode, or GPT4 Only Mode.
The user can specify the number of times to run in continuous mode and which memory backend to use.
The user can allow dangerous file downloads, specify the web browser to use, and suppress output of latest news on startup.
The user can install external dependencies for 3rd party plugins.
Functions:

main(ctx, continuous, continuous_limit, ai_settings, project_dir, skip_reprompt, speak, debug, gpt3only, gpt4only, memory_type, browser_name, allow_downloads, skip_news, workspace_directory, install_plugin_deps) -> None:
Starts the AutoGPT assistant with the specified options.
Classes:

None
Global Variables:

None
Dependencies:

click: A third-party module for creating command-line interfaces.
autogpt.main: A module containing the run_auto_gpt() function which starts the AutoGPT assistant.
"""
import click
from autogpt.config.config import Config

CFG = Config()

@click.group(invoke_without_command=True)
@click.option("-c", "--continuous-mode", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(# TODO TO BE REMOVE
    "--ai-settings", 
    "-C",
    default=CFG.ai_settings_file, 
    help="Specifies which ai_settings.yaml file to use, will also automatically skip the re-prompt.",
)

@click.option( 
    "--project-dir",
    "-D",
    default=CFG.project_dir,
    help="Specifies which directory contains projects.",
)
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("-speak", "--speak-mode", is_flag=True, help="Enable Speak Mode")
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
@click.option(
    # TODO: this is a hidden option for now, necessary for integration testing.
    #   We should make this public once we're ready to roll out agent specific workspaces.
    "--workspace-directory",
    "-w",
    type=click.Path(),
    hidden=True,
)
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.pass_context
def main(
    ctx: click.Context,
    continuous_mode: bool,
    continuous_limit: int,
    ai_settings: str,
    project_dir:str,
    skip_reprompt: bool,
    speak_mode: bool,
    debug: bool,
    gpt3only: bool,
    gpt4only: bool,
    memory_type: str,
    browser_name: str,
    allow_downloads: bool,
    skip_news: bool,
    workspace_directory: str,
    install_plugin_deps: bool,
) -> None:
    """
    Welcome to AutoGPT an experimental open-source application showcasing the capabilities of the GPT-4 pushing the boundaries of AI.

    Start an Auto-GPT assistant.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.main import run_auto_gpt

    if ctx.invoked_subcommand is None:
        run_auto_gpt(
            continuous_mode,
            continuous_limit,
            ai_settings,
            project_dir,
            skip_reprompt,
            speak_mode,
            debug,
            gpt3only,
            gpt4only,
            memory_type,
            browser_name,
            allow_downloads,
            skip_news,
            workspace_directory,
            install_plugin_deps,
        )


CFG= Config()
if __name__ == "__main__":
    main()