"""Marks the functions the database manager serves to processes that hold no
Prisma connection. The manager serves what the modules it is given declare
(``DB_ENDPOINT_MODULES`` in ``db_manager.py``), each under its function name."""

from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_DECLARED = "__db_endpoint__"


def db_endpoint(func: F) -> F:
    """Declare ``func`` a database manager route. Its arguments and result
    cross the RPC as JSON, so it takes no transaction client."""
    setattr(func, _DECLARED, True)
    return func


def declared_endpoints(module: ModuleType) -> dict[str, Callable[..., Any]]:
    """Route name to function, for the endpoints ``module`` itself defines."""
    return {
        name: member
        for name, member in vars(module).items()
        if getattr(member, _DECLARED, False)
        # Not one it imported from another module, nor an alias of its own.
        and member.__module__ == module.__name__ and member.__name__ == name
    }
