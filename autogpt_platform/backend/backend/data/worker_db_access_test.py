"""Code that runs in a worker process reaches the database over RPC.

Only the REST API, the WebSocket API and the DatabaseManager connect Prisma.
The copilot executor and the graph executor do not, so a query function
imported straight into the code they run raises ``ClientNotConnectedError``
in production while passing every test that has a database. The packages
below import query functions only through ``backend.data.db_accessors``.

Only what these packages import themselves is checked, not the helpers they
call in other packages.
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
    "bounded_transaction",
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
        or "test" in path.relative_to(_BACKEND).parts
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


def _bindings(imports: list[tuple[int, str, str]]) -> dict[str, str]:
    """What each name an import binds refers to; ``import a.b`` binds ``a``."""
    bound = {}
    for _, dotted, local in imports:
        root = local.split(".")[0]
        bound[root] = root if "." in local else dotted
    return bound


def _resolve(node: ast.AST, bound: dict[str, str]) -> str | None:
    """The dotted name an attribute chain refers to, if it starts at an import."""
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in bound:
        return ".".join([bound[node.id], *reversed(attrs)])
    return None


def _served_functions() -> set[str]:
    """Dotted names of the functions DatabaseManager exposes over RPC."""
    path = _BACKEND / "data" / "db_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = _bindings(_imports(tree, path))
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound[node.name] = f"backend.data.db_manager.{node.name}"
    served = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_"
            and node.args
        ):
            target = _resolve(node.args[0], bound)
            if target:
                served.add(target)
    return served


_SERVED = _served_functions()


def _violations(module: pathlib.Path) -> list[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    where = module.relative_to(_BACKEND).as_posix()
    found = []
    imports = _imports(tree, module)
    for line, dotted, _ in imports:
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
    bound = _bindings(imports)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "prisma":
                found.append(f"{where}:{node.lineno} calls .prisma()")
        if not isinstance(node, ast.Attribute):
            continue
        target = _resolve(node, bound) or ""
        source, _, name = target.rpartition(".")
        if source == "backend.data.db" and name in _CONNECTION_NAMES:
            found.append(f"{where}:{node.lineno} uses {target}")
        elif target in _SERVED - _PURE:
            found.append(f"{where}:{node.lineno} calls {target}")
    return found


def test_query_modules_are_recognised():
    assert _is_query_module("backend.copilot.db")
    assert not _is_query_module("backend.data.db_accessors")
    assert not _is_query_module("backend.copilot.model")


def test_database_manager_endpoints_are_found():
    assert "backend.copilot.db.get_chat_session_metadata" in _SERVED
    assert len(_SERVED) > 300


def test_worker_packages_do_not_query_the_database_directly():
    modules = _modules()
    assert len(modules) > 100
    found = [line for module in modules for line in _violations(module)]
    if found:
        pytest.fail("\n".join([*found, "", _ADVICE]))


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "from backend.copilot.db import list_recent_chat_sessions\n",
            "imports backend.copilot.db.list_recent_chat_sessions",
        ),
        ("from backend.copilot import db\n", "query module"),
        (
            "from backend.data.user import get_user_by_id\n",
            "imports backend.data.user.get_user_by_id",
        ),
        (
            "from backend.data import user\nasync def f():\n"
            "    await user.get_user_by_id('u')\n",
            "calls backend.data.user.get_user_by_id",
        ),
        ("from backend.data.db import prisma\n", "from backend.data.db"),
        (
            "from backend.data import db as core\nasync def f():\n"
            "    await core.prisma.user.find_first()\n",
            "uses backend.data.db.prisma",
        ),
        ("async def f():\n    await User.prisma().find_first()\n", ".prisma()"),
        (
            "import backend.data.user\nasync def f():\n"
            "    await backend.data.user.get_user_by_id('u')\n",
            "calls backend.data.user.get_user_by_id",
        ),
        (
            "from backend.data.db_manager import _spend_credits\n",
            "imports backend.data.db_manager._spend_credits",
        ),
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
    monkeypatch.setattr(
        "backend.data.worker_db_access_test._is_query_module",
        lambda dotted: dotted == "backend.copilot.db",
    )
    monkeypatch.setattr("backend.data.worker_db_access_test._BACKEND", tmp_path)
    assert _violations(module) == []
