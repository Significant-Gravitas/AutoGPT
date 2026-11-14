"""The shim's waits keep a cancel that lands just as the work they wait for ends.

Python 3.11's asyncio.wait_for returns that work's result instead, and the
cancel is lost: the task goes on as if it hadn't been cancelled. A session
revoked then left a command it ran recorded as done and running (twice: as
its exit was seen, then as its output was read), and a local model's answer
streaming on. asyncio.timeout keeps the cancel, and Python 3.12's wait_for is
built on it. wait_for stays only where a cancel it drops lets a stop finish."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Where asyncio.wait_for stays, by file and function, and why a cancel it
# drops does no harm there.
ALLOWED = {
    ("autogpt_local_executor/handlers.py", "CommandHandler._stop_cancelled"): (
        "a cancelled command's stop, a task of its own (_finish_then_cancel) that only the "
        "event loop's end cancels: a dropped cancel lets it write the command's record"
    ),
    ("autogpt_local_executor/computer_use/mcp_client.py", "StdioMcpClient.close"): (
        "the driver's stop, in a task of its own or just before a raise: a dropped cancel "
        "lets the stop finish"
    ),
    ("autogpt_local_executor/computer_use/mcp_client.py", "StdioMcpClient._read_stdout"): (
        "the reader's look at the exit code once the driver has gone: a dropped cancel lets "
        "it fail the requests still waiting, as close() does next"
    ),
}


def _wait_fors(tree: ast.Module) -> Iterator[tuple[str, int]]:
    """Each use of asyncio.wait_for in `tree`, as the function it is in and
    its line: through the module, under whatever name it was imported as."""
    modules = {"asyncio"}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {
                alias.asname or "asyncio" for alias in node.names if alias.name == "asyncio"
            }
        elif isinstance(node, ast.ImportFrom) and node.module in ("asyncio", "asyncio.tasks"):
            names |= {
                alias.asname or alias.name for alias in node.names if alias.name == "wait_for"
            }
    owners = {*modules, *(f"{module}.tasks" for module in modules)}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> Iterator[tuple[str, int]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                yield from visit(child, (*scope, child.name))
                continue
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "wait_for"
                and ast.unparse(child.value) in owners
            ) or (isinstance(child, ast.Name) and child.id in names):
                yield ".".join(scope), child.lineno
            yield from visit(child, scope)

    yield from visit(tree, ())


def _found() -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted((ROOT / "autogpt_local_executor").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for function, line in _wait_fors(ast.parse(path.read_bytes(), str(path))):
            found.setdefault((relative, function), []).append(line)
    return found


def test_every_wait_keeps_a_cancel_that_lands_as_its_work_ends() -> None:
    dropping = [
        f"{path}:{line}: asyncio.wait_for in {function or 'the module'}"
        for (path, function), lines in sorted(_found().items())
        if (path, function) not in ALLOWED
        for line in lines
    ]
    assert not dropping, (
        "Wait under asyncio.timeout: Python 3.11's asyncio.wait_for drops a cancel that lands "
        "as the work ends. Where that is harmless, add the function to ALLOWED, saying why:\n"
        + "\n".join(dropping)
    )


def test_every_wait_for_allowed_is_still_there() -> None:
    assert sorted(set(ALLOWED) - set(_found())) == []


def test_the_check_finds_wait_for_under_any_name() -> None:
    source = """\
import asyncio
import asyncio as aio
from asyncio import wait_for as bounded
async def a():
    await asyncio.wait_for(work, 1)
class B:
    async def c(self):
        await aio.wait_for(work, 1)
        await asyncio.tasks.wait_for(work, 1)
        await bounded(work, 1)
        await condition.wait_for(ready)
        async with asyncio.timeout(1):
            await work
"""
    found = sorted(_wait_fors(ast.parse(source)))

    assert found == [("B.c", 8), ("B.c", 9), ("B.c", 10), ("a", 5)]
