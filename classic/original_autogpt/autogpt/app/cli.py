"""Main script for the autogpt package."""
from logging import _nameToLevel as logLevelMap
from pathlib import Path
from typing import Optional

import click
from forge.logging.config import LogFormatName

from .telemetry import setup_telemetry


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    setup_telemetry()

    # Invoke `run` by default
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.option("-c", "--continuous", is_flag=True, help="Enable Continuous Mode")
@click.option(
    "-l",
    "--continuous-limit",
    type=int,
    help="Defines the number of times to run in continuous mode",
)
@click.option("--speak", is_flag=True, help="Enable Speak Mode")
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--skip-news",
    is_flag=True,
    help="Specifies whether to suppress the output of latest news on startup.",
)
@click.option(
    "--skip-reprompt",
    "-y",
    is_flag=True,
    help="Skips the re-prompting messages at the beginning of the script",
)
@click.option(
    "--ai-name",
    type=str,
    help="AI name override",
)
@click.option(
    "--ai-role",
    type=str,
    help="AI role override",
)
@click.option(
    "--constraint",
    type=str,
    multiple=True,
    help=(
        "Add or override AI constraints to include in the prompt;"
        " may be used multiple times to pass multiple constraints"
    ),
)
@click.option(
    "--resource",
    type=str,
    multiple=True,
    help=(
        "Add or override AI resources to include in the prompt;"
        " may be used multiple times to pass multiple resources"
    ),
)
@click.option(
    "--best-practice",
    type=str,
    multiple=True,
    help=(
        "Add or override AI best practices to include in the prompt;"
        " may be used multiple times to pass multiple best practices"
    ),
)
@click.option(
    "--override-directives",
    is_flag=True,
    help=(
        "If specified, --constraint, --resource and --best-practice will override"
        " the AI's directives instead of being appended to them"
    ),
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--component-config-file",
    help="Path to a json configuration file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
)
def run(
    continuous: bool,
    continuous_limit: Optional[int],
    speak: bool,
    install_plugin_deps: bool,
    skip_news: bool,
    skip_reprompt: bool,
    ai_name: Optional[str],
    ai_role: Optional[str],
    resource: tuple[str],
    constraint: tuple[str],
    best_practice: tuple[str],
    override_directives: bool,
    debug: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    log_file_format: Optional[str],
    component_config_file: Optional[Path],
) -> None:
    """
    Sets up and runs an agent, based on the task specified by the user, or resumes an
    existing agent.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.app.main import run_auto_gpt

    run_auto_gpt(
        continuous=continuous,
        continuous_limit=continuous_limit,
        skip_reprompt=skip_reprompt,
        speak=speak,
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        skip_news=skip_news,
        install_plugin_deps=install_plugin_deps,
        override_ai_name=ai_name,
        override_ai_role=ai_role,
        resources=list(resource),
        constraints=list(constraint),
        best_practices=list(best_practice),
        override_directives=override_directives,
        component_config_file=component_config_file,
    )


@cli.command()
@click.option(
    "--install-plugin-deps",
    is_flag=True,
    help="Installs external dependencies for 3rd party plugins.",
)
@click.option(
    "--debug", is_flag=True, help="Implies --log-level=DEBUG --log-format=debug"
)
@click.option("--log-level", type=click.Choice([*logLevelMap.keys()]))
@click.option(
    "--log-format",
    help=(
        "Choose a log format; defaults to 'simple'."
        " Also implies --log-file-format, unless it is specified explicitly."
        " Using the 'structured_google_cloud' format disables log file output."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
@click.option(
    "--log-file-format",
    help=(
        "Override the format used for the log file output."
        " Defaults to the application's global --log-format."
    ),
    type=click.Choice([i.value for i in LogFormatName]),
)
def serve(
    install_plugin_deps: bool,
    debug: bool,
    log_level: Optional[str],
    log_format: Optional[str],
    log_file_format: Optional[str],
) -> None:
    """
    Starts an Agent Protocol compliant AutoGPT server, which creates a custom agent for
    every task.
    """
    # Put imports inside function to avoid importing everything when starting the CLI
    from autogpt.app.main import run_auto_gpt_server

    run_auto_gpt_server(
        debug=debug,
        log_level=log_level,
        log_format=log_format,
        log_file_format=log_file_format,
        install_plugin_deps=install_plugin_deps,
    )


if __name__ == "__main__":
    cli()
