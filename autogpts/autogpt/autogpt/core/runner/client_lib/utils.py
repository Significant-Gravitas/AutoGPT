import asyncio
import functools
from bdb import BdbQuit
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

import click

P = ParamSpec("P")
T = TypeVar("T")


def handle_exceptions(
    application_main: Callable[P, T],
    with_debugger: bool,
) -> Callable[P, T]:
    """Wraps a function so that it drops a user into a debugger if it raises an error.

    This is intended to be used as a wrapper for the main function of a CLI application.
    It will catch all errors and drop a user into a debugger if the error is not a
    `KeyboardInterrupt`. If the error is a `KeyboardInterrupt`, it will raise the error.
    If the error is not a `KeyboardInterrupt`, it will log the error and drop a user
    into a debugger if `with_debugger` is `True`.
    If `with_debugger` is `False`, it will raise the error.

    Parameters
    ----------
    application_main
        The function to wrap.
    with_debugger
        Whether to drop a user into a debugger if an error is raised.

    Returns
    -------
    Callable
        The wrapped function.

    """

    @functools.wraps(application_main)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await application_main(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt, click.Abort):
            raise
        except Exception as e:
            if with_debugger:
                print(f"Uncaught exception {e}")
                import pdb

                pdb.post_mortem()
            else:
                raise

    return wrapped


def coroutine(f: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper
