import pathlib

import click

DEFAULT_SETTINGS_FILE = str(pathlib.Path("~/auto-gpt/settings.yaml").expanduser())


@click.command()
@click.option(
    "--settings-file",
    type=click.Path(),
    default=DEFAULT_SETTINGS_FILE,
)
def make_settings(settings_file: str) -> None:
    from autogpt.core.runner.app_lib.settings import make_default_settings

    make_default_settings(pathlib.Path(settings_file))
