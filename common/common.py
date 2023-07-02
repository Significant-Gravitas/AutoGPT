import hashlib
import logging
import os
import traceback
from ast import List
from typing import Any, Iterable, Set, Type

default_not_detailed_errors: Set = set()

# Adapted from compare-my-stocks repo by the author
# https://github.com/eyalk11/compare-my-stocks
# originally used pyro5 error handling to display very detailed exceptions


def simple_exception_handling(
    err_description=None,
    return_on_exc=None,
    lambda_on_exc=None,
    never_throw=False,
    always_throw=False,
    debug=False,
    detailed=True,
    err_to_throw=[],
    prioritize_raise_in_debug=True,
    fill_in_default_desciption=True,
    log_if_thrown=False,
):
    """
    Decorator to handle exceptions. Catches the exception in the decorated function and display proper message.
    If you are debugging, it doesn't catch the error by default allowing you to see it in frame.
    You can also choose to run lambda or return value in case of exception.

    Parameters:
    err_description: Description of the error.
    return_on_exc: Value to return when there is an exception.
    lambda_on_exc: Lambda to execute when there is an exception (prioritized over return).
    never_throw: Never throw the exception.
    always_throw: Always throw the exception.
    debug: Log the error as debug.
    detailed: Log the error in detail.
    err_to_ignore: List of errors to throw.
    prioritize_raise_in_debug: to prioritize raising when debuging even if return_on_exc is set
    fill_in_default_desciption: in case there is no description
    log_if_thrown : if class in err_to_throw , should I throw
    """

    def decorated(func):
        def internal(*args, **kwargs):
            raiseinplace = os.environ.get("PYCHARM_HOSTED") == "1"
            raiseinplace = raiseinplace or any(
                [("VSCODE_" in x) for x in list(os.environ.keys())]
            )  # raise it if being debugged

            if (
                raiseinplace
                and not never_throw
                and (
                    (lambda_on_exc is None and return_on_exc is None)
                    or prioritize_raise_in_debug
                )
            ):
                return func(*args, **kwargs)
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if e.__class__ in err_to_throw and not (log_if_thrown):
                        raise e

                    logf = logging.debug if debug else logging.error
                    if (err_description is None) and fill_in_default_desciption:
                        err_description = f"Exception in {func}"
                    if err_description:
                        logf(err_description)

                    elif e.__class__ not in default_not_detailed_errors and detailed:
                        logf(traceback.format_exc())
                    else:
                        logf(str(e))

                    if always_throw or (e.__class__ in err_to_throw):
                        raise e
                    if lambda_on_exc:
                        return lambda_on_exc()
                    return return_on_exc

        return internal

    return decorated


def calculate_sha256(stream: Iterable) -> str:
    sha256_hash = hashlib.sha256()
    for chunk in stream:
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def is_valid_int(value: str) -> bool:
    """Check if the value is a valid integer

    Args:
        value (str): The value to check

    Returns:
        bool: True if the value is a valid integer, False otherwise
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
