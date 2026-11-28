import types
from pathlib import Path

import pytest

import backend
from backend.data.db_endpoints import db_endpoint, declared_endpoints
from backend.data.db_manager import DB_ENDPOINT_MODULES, DatabaseManager
from backend.util.service import EXPOSED_FLAG


def test_a_module_declares_only_the_endpoints_it_defines_itself():
    lender, module = _module("lender"), _module("borrower")

    @db_endpoint
    async def lent() -> None: ...

    @db_endpoint
    async def served() -> None: ...

    async def private() -> None: ...

    lent.__module__ = lender.__name__
    for function in (served, private):
        function.__module__ = module.__name__
    vars(module).update(lent=lent, served=served, alias=served, private=private)

    assert declared_endpoints(module) == {"served": served}


def test_the_manager_refuses_a_second_endpoint_of_the_same_name():
    """Routes share one namespace; a later one would silently replace the first."""
    module = _module("duplicate")

    @db_endpoint
    async def get_graph() -> None: ...

    get_graph.__module__ = module.__name__
    module.get_graph = get_graph

    with pytest.raises(RuntimeError, match="already has a 'get_graph'"):
        DatabaseManager.serve(module)


def test_every_declared_endpoint_is_served():
    served = {
        name
        for name, attr in vars(DatabaseManager).items()
        if getattr(attr, EXPOSED_FLAG, False)
    }
    declared = {
        route for module in DB_ENDPOINT_MODULES for route in declared_endpoints(module)
    }

    assert declared and declared <= served


def test_every_module_that_declares_an_endpoint_is_one_the_manager_serves():
    """A module the manager was not given is never read: its endpoints would
    work in the REST server and fail in every process that crosses the RPC."""
    root = Path(backend.__file__).parent
    declaring = {
        ".".join(("backend", *path.relative_to(root).with_suffix("").parts))
        for path in root.rglob("*.py")
        if not path.name.endswith("_test.py") and "\n@db_endpoint\n" in path.read_text()
    }

    assert declaring == {module.__name__ for module in DB_ENDPOINT_MODULES}


def _module(name: str) -> types.ModuleType:
    return types.ModuleType(f"backend.data.db_endpoints_test_{name}")
