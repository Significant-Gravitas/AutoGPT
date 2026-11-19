"""Inside a transaction, every query goes through the transaction's client.

The rule covers the body of every ``async with transaction() / .tx()`` block
and of every function that takes the transaction as a ``Prisma`` argument. It
follows calls into ``backend`` functions however they are imported (absolute,
relative, or through a module alias) and however deep. A parameter annotated
``Prisma`` or ``Prisma | None`` is that function's client: handed the
transaction, its ``if client is None`` branches are skipped; handed nothing,
they are checked; handed ``None``, the call itself is refused.
"""

import ast
import functools
import pathlib
from collections.abc import Iterator

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = BACKEND_ROOT.parent
_TRANSACTION_OPENERS = frozenset({"transaction", "tx"})
_RAW_HELPERS = frozenset({"query_raw_with_schema", "execute_raw_with_schema"})
_CLIENT_ANNOTATIONS = frozenset(
    {
        "Prisma",
        "prisma.Prisma",
        "Prisma | None",
        "prisma.Prisma | None",
        "Optional[Prisma]",
        "Optional[prisma.Prisma]",
    }
)
_REQUIRED_CLIENT_ANNOTATIONS = frozenset({"Prisma", "prisma.Prisma"})
_GLOBAL_CLIENT = "backend.data.db.prisma"

Function = ast.AsyncFunctionDef | ast.FunctionDef


def test_transactions_query_only_through_their_own_client():
    offences = [
        offence
        for path in sorted(BACKEND_ROOT.rglob("*.py"))
        if not path.name.endswith("_test.py")
        for offence in transaction_scope_offences(path)
    ]
    assert not offences, (
        "Queries inside a transaction must use its client, e.g. "
        "`Model.prisma(tx)` or a helper handed `tx`:\n" + "\n".join(offences)
    )


def test_the_rule_follows_every_way_a_helper_is_reached(tmp_path):
    package = tmp_path / "backend" / "pkg"
    package.mkdir(parents=True)
    (tmp_path / "backend" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "store.py").write_text(
        "from backend.data.db import prisma\n"
        "\n"
        "async def members(team):\n"
        "    return await prisma.teammember.find_many(where={'teamId': team})\n"
        "\n"
        "async def lookup(name):\n"
        "    return await Folder.prisma().find_first(where={'name': name})\n"
        "\n"
        "async def indirect(name):\n"
        "    return await lookup(name)\n"
        "\n"
        "async def scoped(name, *, client: Prisma | None):\n"
        "    if client is None:\n"
        "        return await lookup(name)\n"
        "    return await Folder.prisma(client).find_first(where={'name': name})\n"
        "\n"
        "async def careless(name, client: Prisma | None = None):\n"
        "    return await Folder.prisma().find_first(where={'name': name})\n"
        "\n"
        "async def ping(name):\n"
        "    await pong(name)\n"
        "    return await Folder.prisma().find_first(where={'name': name})\n"
        "\n"
        "async def pong(name):\n"
        "    return await ping(name)\n"
    )
    (package / "writer.py").write_text(
        "from . import store as store_db\n"
        "from .store import careless, indirect, scoped\n"
        "\n"
        "async def write(name, parent):\n"
        "    async with transaction() as tx:\n"
        "        await store_db.lookup(name)\n"
        "        await indirect(name)\n"
        "        await scoped(name, client=tx)\n"
        "        await scoped(name, client=None)\n"
        "        await careless(name, client=tx)\n"
        "        await query_raw_with_schema('SELECT 1')\n"
        "        if parent is not None:\n"
        "            await store_db.lookup(parent)\n"
        "        await store_db.members(name)\n"
        "        if tx is None and parent is None:\n"
        "            await store_db.lookup(name)\n"
        "        await store_db.ping(name)\n"
        "        await store_db.pong(name)\n"
        "        await Folder.prisma(tx).create(data={'name': name})\n"
    )

    offences = transaction_scope_offences(package / "writer.py", root=tmp_path)

    assert offences == [
        "backend/pkg/writer.py:6 calls lookup, which queries the global client",
        "backend/pkg/writer.py:7 calls indirect, which queries the global client",
        "backend/pkg/writer.py:9 passes None as the client of scoped",
        "backend/pkg/writer.py:10 calls careless, which queries the global client",
        "backend/pkg/writer.py:11 calls query_raw_with_schema without client=",
        "backend/pkg/writer.py:13 calls lookup, which queries the global client",
        "backend/pkg/writer.py:14 calls members, which queries the global client",
        "backend/pkg/writer.py:17 calls ping, which queries the global client",
        "backend/pkg/writer.py:18 calls pong, which queries the global client",
    ]


def transaction_scope_offences(
    path: pathlib.Path, root: pathlib.Path = PACKAGE_ROOT
) -> list[str]:
    module = _module_name(path, root)
    offences: dict[int, str] = {}
    for body, bound in _transaction_scopes(_parse(path)):
        for line, reason in _scan(module, body, bound, frozenset(), root):
            offences.setdefault(line, f"{path.relative_to(root)}:{line} {reason}")
    return [offences[line] for line in sorted(offences)]


def _transaction_scopes(
    tree: ast.Module,
) -> Iterator[tuple[list[ast.stmt], frozenset[str]]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncWith):
            names = frozenset(
                item.optional_vars.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and _call_name(item.context_expr) in _TRANSACTION_OPENERS
                and isinstance(item.optional_vars, ast.Name)
            )
            if names:
                yield node.body, names
        elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            required = frozenset(
                arg.arg
                for arg in _params(node)
                if _annotation(arg) in _REQUIRED_CLIENT_ANNOTATIONS
            )
            if required:
                yield node.body, required


def _scan(
    module: str,
    body: list[ast.stmt],
    bound: frozenset[str],
    unset: frozenset[str],
    root: pathlib.Path,
) -> Iterator[tuple[int, str]]:
    """Each line of *body* that reaches the global client. *bound* names carry
    the transaction; *unset* names are client parameters left at None."""
    for node in _reachable(body, bound, unset):
        if not isinstance(node, ast.Call):
            continue
        if _queries_global(node, unset) or _on_global_client(module, node, root):
            yield node.lineno, "queries the global client"
            continue
        name = _call_name(node)
        if name in _RAW_HELPERS:
            client = next((k.value for k in node.keywords if k.arg == "client"), None)
            if client is None or _is_none(client) or _carries(client, unset):
                yield node.lineno, f"calls {name} without client="
            continue
        target = _resolve_call(module, node, root)
        if target is None:
            continue
        target_module, function = target
        clients = [arg.arg for arg in _params(function) if _is_client_param(arg)]
        passed = _arguments(node, function)
        if any(_is_none(passed.get(c)) for c in clients):
            yield node.lineno, f"passes None as the client of {function.name}"
            continue
        callee_bound = frozenset(c for c in clients if _carries(passed.get(c), bound))
        callee_unset = frozenset(
            c for c in clients if c not in passed or _carries(passed[c], unset)
        )
        if _reaches_global(
            target_module, function.name, callee_bound, callee_unset, root
        ):
            yield node.lineno, f"calls {function.name}, which queries the global client"


_Key = tuple[str, str, frozenset[str], frozenset[str], pathlib.Path]
_REACHES: dict[_Key, bool] = {}
_VISITING: set[_Key] = set()
_CYCLES_CUT = [0]


def _reaches_global(
    module: str,
    function_name: str,
    bound: frozenset[str],
    unset: frozenset[str],
    root: pathlib.Path,
) -> bool:
    key = (module, function_name, bound, unset, root)
    if key in _REACHES:
        return _REACHES[key]
    function = _module_functions(module, root).get(function_name)
    if function is None:
        return False
    if key in _VISITING:
        _CYCLES_CUT[0] += 1
        return False
    cuts_before = _CYCLES_CUT[0]
    _VISITING.add(key)
    try:
        reaches = any(True for _ in _scan(module, function.body, bound, unset, root))
    finally:
        _VISITING.discard(key)
    # A False found while a cycle was cut short may be wrong; only remember it
    # once nothing below was cut.
    if reaches or _CYCLES_CUT[0] == cuts_before:
        _REACHES[key] = reaches
    return reaches


def _reachable(
    body: list[ast.stmt], bound: frozenset[str], unset: frozenset[str]
) -> Iterator[ast.AST]:
    """Nodes of *body*, skipping the branch a client's None test rules out."""
    stack: list[ast.AST] = list(reversed(body))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.If, ast.IfExp)):
            outcome = _test_outcome(node.test, bound, unset)
            if outcome is not None:
                kept = node.body if outcome else node.orelse
                stack.extend(reversed(kept if isinstance(kept, list) else [kept]))
                continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _test_outcome(
    test: ast.expr, bound: frozenset[str], unset: frozenset[str]
) -> bool | None:
    """Whether *test* holds given which clients are set, or None if that
    does not decide it. An `and` is decided False by any one false part."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        parts = [_test_outcome(value, bound, unset) for value in test.values]
        return False if False in parts else None
    none_test = _none_test(test)
    if none_test is None or none_test[0] not in bound | unset:
        return None
    name, holds_if_none = none_test
    return (name in unset) == holds_if_none


def _none_test(test: ast.expr) -> tuple[str, bool] | None:
    """(name, True) for `name is None` / `not name`, (name, False) for
    `name is not None` / `name`."""
    if isinstance(test, ast.Name):
        return test.id, False
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
    ):
        return test.operand.id, True
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.left, ast.Name)
        and _is_none(test.comparators[0])
    ):
        if isinstance(test.ops[0], ast.Is):
            return test.left.id, True
        if isinstance(test.ops[0], ast.IsNot):
            return test.left.id, False
    return None


def _queries_global(node: ast.Call, unset: frozenset[str]) -> bool:
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "prisma"):
        return False
    if not node.args and not node.keywords:
        return True
    client = node.args[0] if node.args else node.keywords[0].value
    return _is_none(client) or _carries(client, unset)


def _on_global_client(module: str, node: ast.Call, root: pathlib.Path) -> bool:
    """`prisma.<model>.<query>(...)` or `db.prisma.<query>(...)` on the
    process-wide client from `backend.data.db`."""
    chain: list[str] = []
    value = node.func
    while isinstance(value, ast.Attribute):
        chain.append(value.attr)
        value = value.value
    if not chain or not isinstance(value, ast.Name):
        return False
    target = _imports(module, root).get(value.id)
    if module == "backend.data.db" and value.id == "prisma":
        target = _GLOBAL_CLIENT
    return target == _GLOBAL_CLIENT or (
        target == "backend.data.db" and chain[-1] == "prisma"
    )


def _carries(value: ast.expr | None, names: frozenset[str]) -> bool:
    return isinstance(value, ast.Name) and value.id in names


def _is_none(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def _is_client_param(arg: ast.arg) -> bool:
    return _annotation(arg) in _CLIENT_ANNOTATIONS


def _annotation(arg: ast.arg) -> str:
    return ast.unparse(arg.annotation) if arg.annotation is not None else ""


def _params(function: Function) -> list[ast.arg]:
    return function.args.posonlyargs + function.args.args + function.args.kwonlyargs


def _arguments(call: ast.Call, function: Function) -> dict[str, ast.expr]:
    positional = function.args.posonlyargs + function.args.args
    passed = {arg.arg: value for arg, value in zip(positional, call.args)}
    passed.update({k.arg: k.value for k in call.keywords if k.arg})
    return passed


def _resolve_call(
    module: str, call: ast.Call, root: pathlib.Path
) -> tuple[str, Function] | None:
    if isinstance(call.func, ast.Name):
        return _resolve_name(module, call.func.id, root)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        target = _imports(module, root).get(call.func.value.id)
        if target and _module_path(target, root):
            return _resolve_name(target, call.func.attr, root)
    return None


def _resolve_name(
    module: str, name: str, root: pathlib.Path, depth: int = 0
) -> tuple[str, Function] | None:
    function = _module_functions(module, root).get(name)
    if function is not None:
        return module, function
    target = _imports(module, root).get(name)
    if target is None or depth > 5 or _module_path(target, root):
        return None
    source, _, imported = target.rpartition(".")
    return _resolve_name(source, imported, root, depth + 1)


@functools.cache
def _module_functions(module: str, root: pathlib.Path) -> dict[str, Function]:
    path = _module_path(module, root)
    if path is None:
        return {}
    return {
        node.name: node
        for node in _parse(path).body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


@functools.cache
def _imports(module: str, root: pathlib.Path) -> dict[str, str]:
    """Local name -> dotted target (a module, or `module.name`), for imports
    that land inside the ``backend`` package."""
    path = _module_path(module, root)
    if path is None:
        return {}
    package = module.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    names: dict[str, str] = {}
    for node in _parse(path).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("backend") and alias.asname:
                    names[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level:
                base = package[: len(package) - node.level + 1]
                source = ".".join(base + ([source] if source else []))
            if not source.startswith("backend"):
                continue
            for alias in node.names:
                names[alias.asname or alias.name] = f"{source}.{alias.name}"
    return names


@functools.cache
def _module_path(module: str, root: pathlib.Path) -> pathlib.Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


@functools.cache
def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""
