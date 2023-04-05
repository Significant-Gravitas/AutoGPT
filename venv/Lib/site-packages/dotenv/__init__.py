from typing import Any, Optional

from .main import (dotenv_values, find_dotenv, get_key, load_dotenv, set_key,
                   unset_key)


def load_ipython_extension(ipython: Any) -> None:
    from .ipython import load_ipython_extension
    load_ipython_extension(ipython)


def get_cli_string(
    path: Optional[str] = None,
    action: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    quote: Optional[str] = None,
):
    """Returns a string suitable for running as a shell script.

    Useful for converting a arguments passed to a fabric task
    to be passed to a `local` or `run` command.
    """
    command = ['dotenv']
    if quote:
        command.append(f'-q {quote}')
    if path:
        command.append(f'-f {path}')
    if action:
        command.append(action)
        if key:
            command.append(key)
            if value:
                if ' ' in value:
                    command.append(f'"{value}"')
                else:
                    command.append(value)

    return ' '.join(command).strip()


__all__ = ['get_cli_string',
           'load_dotenv',
           'dotenv_values',
           'get_key',
           'set_key',
           'unset_key',
           'find_dotenv',
           'load_ipython_extension']
