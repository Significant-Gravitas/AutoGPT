"""Main script for the autogpt package."""
from pathlib import Path
import shutil
import click

def check_autogpt_command(interactive=True):
    """
    Check for the presence of the autogpt command as a way to gauge whether Auto-GPT is installed.
    """
    if shutil.which("autogpt") is None:
        click.echo("Warning: 'autogpt' command not found. With this command, you can run 'autogpt' from anywhere.")
        if interactive and Path("./autogpt").exists():
            if click.prompt("Would you like try and install it now? (y/n)") == "y":
                import subprocess
                subprocess.run(["pip", "install", "-e", "."])
        else:
            click.echo("You can run 'pip install autogpt' to install the Auto-GPT command.")
    
def check_installation(workspace_directory, interactive=True):
    """
    Checks if Auto-GPT is being invoked from within the Auto-GPT folder or if the autogpt command doesn't exist
    and prompts the user to install it if so.
    """
    check_autogpt_command()
    
    if not workspace_directory:
        # Check for default legacy workspace directory & prompt to migrate
        legacy_workspace_name = "auto_gpt_workspace"
        legacy_workspace_directory = None
        
        if Path(Path.cwd() / legacy_workspace_name).exists():
            legacy_workspace_directory = Path.cwd() / legacy_workspace_name
        elif Path(Path(__file__).parent / legacy_workspace_name).exists():
            legacy_workspace_directory = Path.cwd() / legacy_workspace_name 
        
        if legacy_workspace_directory:
            click.echo("Warning: Old workspace directory found at " + str(legacy_workspace_directory))
            
            if interactive:
                if click.prompt("Would you like to migrate it to your home directory? (y/n)") == "y":
                    new_workspace_directory = Path.home() / legacy_workspace_name
                    click.echo("Migrating workspace directory to " + str(new_workspace_directory))
                    shutil.move(legacy_workspace_directory, new_workspace_directory)
                    click.echo("Workspace directory migrated successfully.")
                    workspace_directory = new_workspace_directory
                else:
                    click.echo("Please move your old workspace directory to another location, for example, your home directory. The old workspace directory will be deprecated in the future.")
                    click.echo("You can also specify a custom workspace directory with the --workspace-directory flag.")
                    workspace_directory = legacy_workspace_directory
    
    if not workspace_directory:
        if interactive:
            # TODO: Kick off interactive install, ask for workspace directory, openai key etc.
            pass  
        else:
            # TODO: Add non-interactive install, set default workspace directory to home directory
            pass



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
    "--prompt-settings",
    "-P",
    help="Specifies which prompt_settings.yaml file to use.",
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
    continuous: bool,
    continuous_limit: int,
    ai_settings: str,
    prompt_settings: str,
    skip_reprompt: bool,
    speak: bool,
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

    check_installation(workspace_directory=workspace_directory, interative=(not continuous))

    if ctx.invoked_subcommand is None:
        run_auto_gpt(
            continuous,
            continuous_limit,
            ai_settings,
            prompt_settings,
            skip_reprompt,
            speak,
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


if __name__ == "__main__":
    main()
