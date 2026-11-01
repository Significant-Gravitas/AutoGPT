"""Inside a transaction, every query goes through the transaction's client,
so a transaction never needs a second connection from the pool it holds one of.

The rule covers the body of every ``async with transaction() / .tx()`` block
and of every function that takes the transaction as a ``Prisma`` argument, and
follows one call into a helper of the same module or of another ``backend``
module.
"""

import ast
import functools
import pathlib

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_ROOT = BACKEND_ROOT.parent
_TRANSACTION_OPENERS = frozenset({"transaction", "tx"})
_RAW_HELPERS = frozenset({"query_raw_with_schema", "execute_raw_with_schema"})


def test_transactions_query_only_through_their_own_client():
    offenders = [
        offence
        for path in sorted(BACKEND_ROOT.rglob("*.py"))
        if not path.name.endswith("_test.py")
        for offence in transaction_scope_offences(path)
    ]
    assert not offenders, (
        "Queries inside a transaction must use its client, e.g. "
        "`Model.prisma(tx)` or a helper that takes `tx`:\n" + "\n".join(offenders)
    )


def test_the_rule_sees_a_helper_on_the_global_client(tmp_path):
    module = tmp_path / "sample.py"
    module.write_text(
        "async def write(name):\n"
        "    async with transaction() as tx:\n"
        "        if await _exists(name):\n"
        "            return\n"
        "        await Folder.prisma(tx).create(data={'name': name})\n"
        "\n"
        "async def _exists(name):\n"
        "    return await Folder.prisma().find_first(where={'name': name})\n"
        "\n"
        "async def _mark(tx: Prisma):\n"
        "    await query_raw_with_schema('SELECT 1')\n"
    )

    offences = transaction_scope_offences(module, root=tmp_path)

    assert offences == [
        "sample.py:3 calls _exists, which queries the global client",
        "sample.py:11 calls query_raw_with_schema without client=",
    ]


def transaction_scope_offences(
    path: pathlib.Path, root: pathlib.Path = PACKAGE_ROOT
) -> list[str]:
    tree = ast.parse(path.read_text())
    module = ".".join(path.relative_to(root).with_suffix("").parts)
    imported = {
        alias.asname or alias.name: (node.module, alias.name)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("backend")
        for alias in node.names
    }
    offences: dict[int, str] = {}
    for scope in _transaction_scopes(tree):
        for node in ast.walk(ast.Module(body=scope, type_ignores=[])):
            if reason := _global_client_use(node, module, imported, root):
                offences[node.lineno] = (
                    f"{path.relative_to(root)}:{node.lineno} {reason}"
                )
    return [offences[line] for line in sorted(offences)]


def _transaction_scopes(tree: ast.Module) -> list[list[ast.stmt]]:
    scopes = [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncWith) and _opens_transaction(node)
    ]
    scopes += [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and _takes_transaction(node)
    ]
    return scopes


def _opens_transaction(node: ast.AsyncWith) -> bool:
    return any(
        isinstance(item.context_expr, ast.Call)
        and item.optional_vars is not None
        and _call_name(item.context_expr) in _TRANSACTION_OPENERS
        for item in node.items
    )


def _takes_transaction(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    return any(
        arg.annotation is not None
        and ast.unparse(arg.annotation) in ("Prisma", "prisma.Prisma")
        for arg in node.args.args + node.args.kwonlyargs
    )


def _global_client_use(
    node: ast.AST,
    module: str,
    imported: dict[str, tuple[str, str]],
    root: pathlib.Path,
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if _is_global_prisma(node):
        return "queries the global client"
    name = _call_name(node)
    if name in _RAW_HELPERS and not any(k.arg == "client" for k in node.keywords):
        return f"calls {name} without client="
    if not isinstance(node.func, ast.Name):
        return None
    target = (module, name)
    if name in imported and name not in _module_functions(module, root):
        target = imported[name]
    if _module_functions(target[0], root).get(target[1]):
        return f"calls {name}, which queries the global client"
    return None


@functools.cache
def _module_functions(module: str, root: pathlib.Path) -> dict[str, bool]:
    path = root.joinpath(*module.split(".")).with_suffix(".py")
    if not path.exists():
        return {}
    return {
        node.name: any(_is_global_prisma(n) for n in ast.walk(node))
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


def _is_global_prisma(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prisma"
        and not node.args
        and not node.keywords
    )


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""
