import sys


def load_contextvar_class():
    if sys.version_info >= (3, 7):
        from contextvars import ContextVar
    elif sys.version_info >= (3, 5, 3):
        from aiocontextvars import ContextVar
    else:
        from contextvars import ContextVar

    return ContextVar


ContextVar = load_contextvar_class()
