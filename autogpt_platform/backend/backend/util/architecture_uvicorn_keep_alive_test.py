"""Every uvicorn server says how long it keeps an idle connection open.

Motivation: uvicorn closes a keep-alive connection after 5 s idle by default,
while the clients and proxies in front of a server keep theirs open longer:
Google's load balancer for 600 s, the RPC clients for 30 s. A request sent on a
connection just as the server closes it gets no answer: the client sees a reset
or "socket hang up", and the load balancer answers 502. So every
``uvicorn.run(...)`` and ``uvicorn.Config(...)`` in product code passes
``timeout_keep_alive``, longer than anything in front of that server keeps a
connection idle: by name, or in the dict literal a backend function returns for
the call to unpack, as ``**rest_api_uvicorn_options(config)`` does.

Test files aren't read: their servers last one test, with nothing in front.
"""

import ast
import textwrap
from collections.abc import Iterator

import pytest

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _is_test_file,
    _iter_backend_py_files,
)

_KEEP_ALIVE = "timeout_keep_alive"
_SERVER_FACTORIES = frozenset({"run", "Config"})

# Module name -> (path to report, parsed source).
_Modules = dict[str, tuple[str, ast.Module]]


def test_every_uvicorn_server_sets_its_keep_alive():
    servers = list(_servers(_backend_modules()))
    assert servers, "found no uvicorn server: the scan no longer recognises them"
    offenders = [f"  {where}" for where, sets_it in servers if not sets_it]
    assert not offenders, (
        "These uvicorn servers keep uvicorn's default 5 s keep-alive, so a "
        "client or proxy that reuses a connection after that gets no answer, "
        "or a 502 from the load balancer. Pass `timeout_keep_alive=` longer "
        "than anything in front of the server keeps a connection idle (the API "
        "and WebSocket servers read `server_keep_alive_timeout_seconds`, the "
        "RPC servers `rpc_server_keepalive_seconds`): by name, or in the dict "
        "literal returned by a function the call unpacks (`**options(config)`, "
        "defined in that module or imported from another by name):\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source, sets_it",
    [
        pytest.param(
            "import uvicorn\nuvicorn.run(app, timeout_keep_alive=60)",
            True,
            id="by-name",
        ),
        pytest.param(
            "import uvicorn\nuvicorn.run(app, port=8006)",
            False,
            id="uvicorns-default",
        ),
        pytest.param(
            "import uvicorn as u\nu.Server(u.Config(app))",
            False,
            id="aliased-module",
        ),
        pytest.param(
            "from uvicorn import Config\nConfig(app, timeout_keep_alive=60)",
            True,
            id="imported-config",
        ),
        pytest.param(
            "from uvicorn import run as serve\nserve(app)",
            False,
            id="renamed-run",
        ),
        pytest.param(
            "import uvicorn\nuvicorn.run(app, **OPTIONS)",
            False,
            id="unpacked-name",
        ),
        pytest.param(
            """
            import uvicorn
            def options():
                return {"port": 8006, "timeout_keep_alive": 620}
            uvicorn.run(app, **options())
            """,
            True,
            id="unpacked-builder",
        ),
        pytest.param(
            """
            import uvicorn
            def options():
                return dict(port=8006, timeout_keep_alive=620)
            uvicorn.run(app, **options())
            """,
            True,
            id="unpacked-dict-call",
        ),
        pytest.param(
            """
            import uvicorn
            def without_it(**kwargs):
                return {"port": 8006}
            def options():
                return without_it(timeout_keep_alive=620)
            uvicorn.run(app, **options())
            """,
            False,
            id="unpacked-call-that-drops-it",
        ),
        pytest.param(
            """
            import uvicorn
            def options():
                return {"port": 8006}
            uvicorn.run(app, **options())
            """,
            False,
            id="unpacked-builder-without-it",
        ),
        pytest.param(
            """
            import uvicorn
            def options(local):
                if local:
                    return {"port": 8006}
                return {"port": 8006, "timeout_keep_alive": 620}
            uvicorn.run(app, **options(True))
            """,
            False,
            id="a-return-without-it",
        ),
    ],
)
def test_the_scan_reads_each_way_of_starting_a_server(source: str, sets_it: bool):
    tree = ast.parse(textwrap.dedent(source))
    [(_, found)] = _servers({"backend.server": ("backend/server.py", tree)})
    assert found is sets_it


@pytest.mark.parametrize(
    "imported, returned, sets_it",
    [
        ("from backend.api.options import options", "{'timeout_keep_alive': 9}", True),
        ("from .options import options", "{'timeout_keep_alive': 9}", True),
        ("from ..api.options import options", "{'timeout_keep_alive': 9}", True),
        ("from backend.api.options import options", "{'port': 8006}", False),
    ],
)
def test_the_scan_follows_a_builder_imported_from_another_module(
    imported: str, returned: str, sets_it: bool
):
    builder = f"def options():\n    return {returned}\n"
    server = f"import uvicorn\n{imported}\nuvicorn.run(app, **options())\n"
    modules = {
        "backend.api.options": ("backend/api/options.py", ast.parse(builder)),
        "backend.api.server": ("backend/api/server.py", ast.parse(server)),
    }
    assert list(_servers(modules)) == [("backend/api/server.py:3", sets_it)]


def _backend_modules() -> _Modules:
    modules: _Modules = {}
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT)
        if _is_test_file(rel):
            continue
        source = py.read_text(encoding="utf-8")
        # Only a file naming one or the other holds a server or its options.
        if "uvicorn" not in source and _KEEP_ALIVE not in source:
            continue
        parts = ("backend", *rel.with_suffix("").parts)
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        tree = ast.parse(source, filename=str(py))
        modules[name] = (f"backend/{rel.as_posix()}", tree)
    return modules


def _servers(modules: _Modules) -> Iterator[tuple[str, bool]]:
    """Each uvicorn server as "path:line", and whether it sets its keep-alive."""
    builders = {
        (name, node.name)
        for name, (_, tree) in modules.items()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _returns_keep_alive(node)
    }
    for name, (path, tree) in modules.items():
        functions = _functions_called_by_name(name, path, tree)
        for call in _uvicorn_servers(tree):
            yield f"{path}:{call.lineno}", any(
                _sets_keep_alive(keyword, functions, builders)
                for keyword in call.keywords
            )


def _uvicorn_servers(tree: ast.Module) -> Iterator[ast.Call]:
    """The `uvicorn.run(...)` and `uvicorn.Config(...)` calls, however uvicorn
    or those names were imported."""
    modules: set[str] = set()
    factories: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or "uvicorn"
                for alias in node.names
                if alias.name.partition(".")[0] == "uvicorn"
            )
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if (node.module or "").partition(".")[0] != "uvicorn":
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                (factories if alias.name in _SERVER_FACTORIES else modules).add(local)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in factories:
            yield node
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in _SERVER_FACTORIES
            and _root_name(func.value) in modules
        ):
            yield node


def _sets_keep_alive(
    keyword: ast.keyword,
    functions: dict[str, tuple[str, str]],
    builders: set[tuple[str, str]],
) -> bool:
    if keyword.arg == _KEEP_ALIVE:
        return True
    unpacked = keyword.value
    return (
        keyword.arg is None
        and isinstance(unpacked, ast.Call)
        and isinstance(unpacked.func, ast.Name)
        and functions.get(unpacked.func.id) in builders
    )


def _functions_called_by_name(
    name: str, path: str, tree: ast.Module
) -> dict[str, tuple[str, str]]:
    """Where each bare name in the module comes from, as (module, name): its
    own functions, and every name it imports by name, function or not. Only
    the builders among them matter to the scan."""
    package = name if path.endswith("__init__.py") else name.rpartition(".")[0]
    functions = {
        alias.asname or alias.name: (_imported_module(package, node), alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    functions.update(
        (node.name, (name, node.name))
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return functions


def _imported_module(package: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - node.level + 1]
    return ".".join([*base, node.module] if node.module else base)


def _returns_keep_alive(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether everything the function returns names the keep-alive."""
    returned = list(_returned(function))
    return bool(returned) and all(_names_keep_alive(value) for value in returned)


def _returned(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.expr | None]:
    """The values of the function's own `return`s, not a nested function's."""
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            yield node.value
        elif not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            stack.extend(ast.iter_child_nodes(node))


def _names_keep_alive(value: ast.expr | None) -> bool:
    if isinstance(value, ast.Dict):
        return any(
            isinstance(key, ast.Constant) and key.value == _KEEP_ALIVE
            for key in value.keys
        )
    # Only `dict(...)` returns its keywords; another function's may not.
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "dict"
    ):
        return any(keyword.arg == _KEEP_ALIVE for keyword in value.keywords)
    return False


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None
