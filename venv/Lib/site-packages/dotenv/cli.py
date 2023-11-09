import json
import os
import shlex
import sys
from contextlib import contextmanager
from subprocess import Popen
from typing import Any, Dict, IO, Iterator, List

try:
    import click
except ImportError:
    sys.stderr.write('It seems python-dotenv is not installed with cli option. \n'
                     'Run pip install "python-dotenv[cli]" to fix this.')
    sys.exit(1)

from .main import dotenv_values, set_key, unset_key
from .version import __version__


def enumerate_env():
    """
    Return a path for the ${pwd}/.env file.

    If pwd does not exist, return None.
    """
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        return None
    path = os.path.join(cwd, '.env')
    return path


@click.group()
@click.option('-f', '--file', default=enumerate_env(),
              type=click.Path(file_okay=True),
              help="Location of the .env file, defaults to .env file in current working directory.")
@click.option('-q', '--quote', default='always',
              type=click.Choice(['always', 'never', 'auto']),
              help="Whether to quote or not the variable values. Default mode is always. This does not affect parsing.")
@click.option('-e', '--export', default=False,
              type=click.BOOL,
              help="Whether to write the dot file as an executable bash script.")
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context, file: Any, quote: Any, export: Any) -> None:
    """This script is used to set, get or unset values from a .env file."""
    ctx.obj = {'QUOTE': quote, 'EXPORT': export, 'FILE': file}


@contextmanager
def stream_file(path: os.PathLike) -> Iterator[IO[str]]:
    """
    Open a file and yield the corresponding (decoded) stream.

    Exits with error code 2 if the file cannot be opened.
    """

    try:
        with open(path) as stream:
            yield stream
    except OSError as exc:
        print(f"Error opening env file: {exc}", file=sys.stderr)
        exit(2)


@cli.command()
@click.pass_context
@click.option('--format', default='simple',
              type=click.Choice(['simple', 'json', 'shell', 'export']),
              help="The format in which to display the list. Default format is simple, "
                   "which displays name=value without quotes.")
def list(ctx: click.Context, format: bool) -> None:
    """Display all the stored key/value."""
    file = ctx.obj['FILE']

    with stream_file(file) as stream:
        values = dotenv_values(stream=stream)

    if format == 'json':
        click.echo(json.dumps(values, indent=2, sort_keys=True))
    else:
        prefix = 'export ' if format == 'export' else ''
        for k in sorted(values):
            v = values[k]
            if v is not None:
                if format in ('export', 'shell'):
                    v = shlex.quote(v)
                click.echo(f'{prefix}{k}={v}')


@cli.command()
@click.pass_context
@click.argument('key', required=True)
@click.argument('value', required=True)
def set(ctx: click.Context, key: Any, value: Any) -> None:
    """Store the given key/value."""
    file = ctx.obj['FILE']
    quote = ctx.obj['QUOTE']
    export = ctx.obj['EXPORT']
    success, key, value = set_key(file, key, value, quote, export)
    if success:
        click.echo(f'{key}={value}')
    else:
        exit(1)


@cli.command()
@click.pass_context
@click.argument('key', required=True)
def get(ctx: click.Context, key: Any) -> None:
    """Retrieve the value for the given key."""
    file = ctx.obj['FILE']

    with stream_file(file) as stream:
        values = dotenv_values(stream=stream)

    stored_value = values.get(key)
    if stored_value:
        click.echo(stored_value)
    else:
        exit(1)


@cli.command()
@click.pass_context
@click.argument('key', required=True)
def unset(ctx: click.Context, key: Any) -> None:
    """Removes the given key."""
    file = ctx.obj['FILE']
    quote = ctx.obj['QUOTE']
    success, key = unset_key(file, key, quote)
    if success:
        click.echo(f"Successfully removed {key}")
    else:
        exit(1)


@cli.command(context_settings={'ignore_unknown_options': True})
@click.pass_context
@click.option(
    "--override/--no-override",
    default=True,
    help="Override variables from the environment file with those from the .env file.",
)
@click.argument('commandline', nargs=-1, type=click.UNPROCESSED)
def run(ctx: click.Context, override: bool, commandline: List[str]) -> None:
    """Run command with environment variables present."""
    file = ctx.obj['FILE']
    if not os.path.isfile(file):
        raise click.BadParameter(
            f'Invalid value for \'-f\' "{file}" does not exist.',
            ctx=ctx
        )
    dotenv_as_dict = {
        k: v
        for (k, v) in dotenv_values(file).items()
        if v is not None and (override or k not in os.environ)
    }

    if not commandline:
        click.echo('No command given.')
        exit(1)
    ret = run_command(commandline, dotenv_as_dict)
    exit(ret)


def run_command(command: List[str], env: Dict[str, str]) -> int:
    """Run command in sub process.

    Runs the command in a sub process with the variables from `env`
    added in the current environment variables.

    Parameters
    ----------
    command: List[str]
        The command and it's parameters
    env: Dict
        The additional environment variables

    Returns
    -------
    int
        The return code of the command

    """
    # copy the current environment variables and add the vales from
    # `env`
    cmd_env = os.environ.copy()
    cmd_env.update(env)

    p = Popen(command,
              universal_newlines=True,
              bufsize=0,
              shell=False,
              env=cmd_env)
    _, _ = p.communicate()

    return p.returncode
