import pathlib

import click

DEFAULT_SETTINGS_FILE = str(
    pathlib.Path("~/auto-gpt/default_agent_settings.yml").expanduser()
)


@click.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
def make_settings(settings_file: str) -> None:
    from autogpt.core.runner.client_lib.settings import make_user_configuration

    make_user_configuration(pathlib.Path(settings_file))
