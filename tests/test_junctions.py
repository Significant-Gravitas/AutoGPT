"""A Windows junction is told apart on every Python the package supports.

`os.path.isjunction` came with Python 3.12, and the package runs on 3.11
too. Where FILE_DELETE asked it directly, a junction on Windows with 3.11
counted as a folder of its own: a recursive delete called `rmtree` on it,
which refuses a junction, and the delete failed.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import autogpt_local_executor
import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.directory_browser import _is_reparse_point
from autogpt_local_executor.handlers import FileHandler
from autogpt_local_executor.protocol import (
    AckMessage,
    FileDeleteMessage,
    FileDeletePayload,
    new_id,
    now_ts,
)

PACKAGE = Path(autogpt_local_executor.__file__).resolve().parent

# The one function that may ask Python whether a path is a junction: it goes
# on to read the reparse tag itself where Python can't say.
FALLBACK = ("directory_browser.py", "_is_reparse_point")
# The names Python 3.12 added for it, on os.path and on Path.
JUNCTION_CALLS = frozenset({"isjunction", "is_junction"})


def _config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
    )


def _junction_asks(tree: ast.AST) -> list[tuple[str | None, int]]:
    """Each place a module names a junction call, as (the function it is in,
    its line): `os.path.isjunction`, `path.is_junction()`, and either name
    as a string, which is how `getattr` asks for it."""
    found: list[tuple[str | None, int]] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        named = (
            node.attr
            if isinstance(node, ast.Attribute)
            else node.id
            if isinstance(node, ast.Name)
            else node.value
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            else None
        )
        if named in JUNCTION_CALLS:
            found.append((function, getattr(node, "lineno", 0)))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def test_only_the_fallback_asks_python_whether_a_path_is_a_junction() -> None:
    """Asked anywhere else, the answer is missing on Python 3.11."""
    elsewhere = [
        f"{path.relative_to(PACKAGE)}:{line} (in {function})"
        for path in sorted(PACKAGE.rglob("*.py"))
        for function, line in _junction_asks(ast.parse(path.read_bytes(), str(path)))
        if (path.name, function) != FALLBACK
    ]

    assert elsewhere == []


def test_the_check_sees_each_way_of_asking() -> None:
    source = (
        "import os\n"
        "def a(p):\n"
        "    return os.path.isjunction(p)\n"
        "def b(p):\n"
        "    return p.is_junction()\n"
        "def c(p):\n"
        "    ask = getattr(os.path, 'isjunction', None)\n"
        "    return ask and ask(p)\n"
        "def d(p):\n"
        "    return os.path.islink(p)\n"
    )

    assert sorted(_junction_asks(ast.parse(source))) == [("a", 3), ("b", 5), ("c", 7)]


def test_the_fallback_does_ask_and_has_its_own_way_too() -> None:
    """The exception above is real: the function is there, asks Python where
    it can, and reads the reparse tag where it can't."""
    browser = PACKAGE / FALLBACK[0]
    asks = _junction_asks(ast.parse(browser.read_bytes(), str(browser)))

    assert [function for function, _line in asks] == [FALLBACK[1]]
    assert "st_reparse_tag" in browser.read_text(encoding="utf-8")


def test_a_junction_is_a_link_where_python_cannot_say(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.11, on any OS: with no `os.path.isjunction`, what tells a
    junction is the package's own look at it."""
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    monkeypatch.setattr(
        handlers_module, "_is_reparse_point", lambda path: Path(path).name == "junction"
    )
    junction = tmp_path / "junction"
    junction.mkdir()
    folder = tmp_path / "folder"
    folder.mkdir()

    assert handlers_module._is_link(junction) is True
    assert handlers_module._is_own_directory(junction) is False
    assert handlers_module._is_link(folder) is False
    assert handlers_module._is_own_directory(folder) is True


def _junction_or_skip(junction: Path, target: Path) -> None:
    """Make `junction` point at `target`, which is beside it. cmd.exe is
    given their two names alone, and run in their folder: given whole paths,
    it read a temporary folder with `&` in its name (`C:\\QA&Test`) as the
    end of the command, made no junction, and both tests here were skipped."""
    assert junction.parent == target.parent
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", junction.name, target.name],
        cwd=junction.parent,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip(f"mklink /J failed here: {made.stdout!r} {made.stderr!r}")


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
@pytest.mark.parametrize("recursive", [False, True])
async def test_delete_removes_a_junction_and_never_what_it_points_to(
    tmp_path: Path, recursive: bool
) -> None:
    """A real junction (`mklink /J`, which needs no privilege), on the
    Python this runs under: 3.11 and 3.12 in CI."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "report.txt").write_text("keep", encoding="utf-8")
    junction = tmp_path / "shortcut"
    _junction_or_skip(junction, documents)
    assert _is_reparse_point(junction) is True
    assert _is_reparse_point(documents) is False
    assert handlers_module._is_link(junction) is True

    response = await FileHandler(_config(tmp_path)).handle_delete(
        FileDeleteMessage(
            id=new_id(),
            ts=now_ts(),
            payload=FileDeletePayload(path=str(junction), recursive=recursive),
        )
    )

    assert isinstance(response, AckMessage), response
    assert not os.path.lexists(junction)
    assert (documents / "report.txt").read_text(encoding="utf-8") == "keep"
