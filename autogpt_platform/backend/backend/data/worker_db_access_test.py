"""Code that runs in a worker process reaches the database over RPC.

Only the REST API, the WebSocket API and the DatabaseManager connect Prisma.
The copilot executor and the graph executor do not, so a query function
imported straight into the code they run raises ``ClientNotConnectedError``
in production while passing every test that has a database. The packages
below import query functions only through ``backend.data.db_accessors``.
"""

import ast
import pathlib

import pytest

_BACKEND = pathlib.Path(__file__).parent.parent

_WORKER_PACKAGES = (
    "blocks",
    "copilot/baseline",
    "copilot/executor",
    "copilot/sdk",
    "copilot/tools",
)

_CONNECTION_NAMES = {
    "connect",
    "execute_raw_with_schema",
    "prisma",
    "query_raw_with_schema",
    "transaction",
}

# Functions in a query module that never touch the database.
_PURE = {"backend.api.features.library.db.collect_tree_ids"}

_ADVICE = (
    "This package runs without a Prisma connection. Call it through "
    "backend.data.db_accessors, adding the function to DatabaseManager and "
    "DatabaseManagerAsyncClient in backend/data/db_manager.py if it is not "
    "there yet."
)


def _is_test_support(path: pathlib.Path) -> bool:
    return (
        path.name.endswith("_test.py")
        or path.name in ("conftest.py", "_test_data.py")
        or "test" in path.parts
    )


def _modules() -> list[pathlib.Path]:
    return sorted(
        path
        for package in _WORKER_PACKAGES
        for path in (_BACKEND / package).rglob("*.py")
        if not _is_test_support(path)
    )


def _imports(tree: ast.AST, module: pathlib.Path) -> list[tuple[int, str, str]]:
    """Every ``(line, dotted name, local name)`` the module imports."""
    package = ("backend", *module.relative_to(_BACKEND).parent.parts)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name, alias.asname or alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = list(package[: len(package) - node.level + 1]) if node.level else []
            source = ".".join(base + ([node.module] if node.module else []))
            for alias in node.names:
                found.append(
                    (node.lineno, f"{source}.{alias.name}", alias.asname or alias.name)
                )
    return found


def _is_query_module(dotted: str) -> bool:
    if not dotted.startswith("backend.") or dotted in (
        "backend.data.db",
        "backend.data.db_accessors",
    ):
        return False
    last = dotted.rsplit(".", 1)[-1]
    if last != "db" and not last.endswith("_db"):
        return False
    path = _BACKEND.joinpath(*dotted.split(".")[1:])
    return path.with_suffix(".py").is_file() or path.is_dir()


def _served_functions() -> set[str]:
    """Dotted names of the functions DatabaseManager exposes over RPC."""
    tree = ast.parse((_BACKEND / "data" / "db_manager.py").read_text(encoding="utf-8"))
    names = {
        local: dotted
        for _, dotted, local in _imports(tree, _BACKEND / "data" / "db_manager.py")
    }
    served = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_"
            and node.args
        ):
            continue
        target, attrs = node.args[0], []
        while isinstance(target, ast.Attribute):
            attrs.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name) and target.id in names:
            served.add(".".join([names[target.id], *reversed(attrs)]))
    return served


_SERVED = _served_functions()


def _violations(module: pathlib.Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    where = module.relative_to(_BACKEND).as_posix()
    found = []
    query_modules = {}
    for line, dotted, local in _imports(tree, module):
        source, _, name = dotted.rpartition(".")
        if dotted in _PURE:
            continue
        if source == "backend.data.db" and name in _CONNECTION_NAMES:
            found.append(f"{where}:{line} imports {name} from backend.data.db")
        elif dotted in _SERVED:
            found.append(f"{where}:{line} imports {dotted}")
        elif _is_query_module(dotted):
            found.append(f"{where}:{line} imports the query module {dotted}")
        elif _is_query_module(source) and not name[:1].isupper():
            found.append(f"{where}:{line} imports {dotted}")
        elif any(served.startswith(f"{dotted}.") for served in _SERVED):
            query_modules[local] = dotted
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "prisma":
                found.append(f"{where}:{node.lineno} calls .prisma()")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and f"{query_modules.get(node.value.id)}.{node.attr}" in _SERVED
        ):
            found.append(
                f"{where}:{node.lineno} calls "
                f"{query_modules[node.value.id]}.{node.attr}"
            )
    return found


def test_database_manager_endpoints_are_found():
    assert "backend.copilot.db.get_chat_session_metadata" in _SERVED
    assert len(_SERVED) > 300


def test_worker_packages_do_not_query_the_database_directly():
    found = [line for module in _modules() for line in _violations(module)]
    if found:
        pytest.fail("\n".join([*found, "", _ADVICE]))


@pytest.mark.parametrize(
    "source, expected",
    [
        ("from backend.copilot.db import list_recent_chat_sessions\n", "imports"),
        ("from backend.copilot import db\n", "query module"),
        ("from backend.data.user import get_user_by_id\n", "imports"),
        (
            "from backend.data import user\nasync def f():\n"
            "    await user.get_user_by_id('u')\n",
            "calls backend.data.user.get_user_by_id",
        ),
        ("from backend.data.db import prisma\n", "from backend.data.db"),
        ("async def f():\n    await User.prisma().find_first()\n", ".prisma()"),
    ],
)
def test_direct_database_access_is_reported(tmp_path, monkeypatch, source, expected):
    package = tmp_path / "copilot" / "tools"
    package.mkdir(parents=True)
    module = package / "example.py"
    module.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        "backend.data.worker_db_access_test._is_query_module",
        lambda dotted: dotted in ("backend.copilot.db",),
    )
    monkeypatch.setattr("backend.data.worker_db_access_test._BACKEND", tmp_path)
    assert expected in "\n".join(_violations(module))


@pytest.mark.parametrize(
    "source",
    [
        "from backend.data.db_accessors import chat_db\n",
        "from backend.copilot.db import PaginatedMessages\n",
        "from backend.data.user import User\n",
    ],
)
def test_routed_access_and_types_pass(tmp_path, monkeypatch, source):
    module = tmp_path / "example.py"
    module.write_text(source, encoding="utf-8")
    monkeypatch.setattr("backend.data.worker_db_access_test._BACKEND", tmp_path)
    assert _violations(module) == []
