"""Every text file the shim and its tests read or write names its encoding.

Without one, Python uses the locale's: cp1252 on most Windows machines, where
a config.toml, an audit log or an autostart entry the shim reads and writes
as UTF-8 then fails to decode, and text cp1252 lacks can't be written at
all. Ruff's PLW1514 checks this, but only as a preview rule in the Ruff
versions the lint jobs run."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Calls that open a file, by name, and where their mode argument is.
OPENERS = {"open": 1, "io.open": 1, "os.fdopen": 1, "codecs.open": 1}
MODE_LETTERS = set("rwxabt+U")


def _modes(node: ast.expr | None) -> set[str] | None:
    """The modes `node` can be (the default where it's left out), or None
    where they can't be read from the source."""
    if node is None:
        return {"r"}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value} if node.value and set(node.value) <= MODE_LETTERS else None
    if isinstance(node, ast.IfExp):
        body, orelse = _modes(node.body), _modes(node.orelse)
        return None if body is None or orelse is None else body | orelse
    return None


def _mode_argument(call: ast.Call, position: int) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value
    return call.args[position] if len(call.args) > position else None


def _text_calls_without_an_encoding(tree: ast.AST) -> Iterator[ast.Call]:
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if any(keyword.arg in ("encoding", None) for keyword in call.keywords):
            continue  # named, or passed on in **kwargs
        func, name = call.func, ast.unparse(call.func)
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            yield call
            continue
        if name in OPENERS:
            position = OPENERS[name]
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            position = 0  # Path.open(mode); another object's open() takes no mode first
        else:
            continue
        modes = _modes(_mode_argument(call, position))
        if modes is not None and any("b" not in mode for mode in modes):
            yield call


def test_every_text_file_read_and_write_names_its_encoding() -> None:
    unnamed = [
        f"{path.relative_to(ROOT)}:{call.lineno}: {ast.unparse(call.func)}(...)"
        for folder in ("autogpt_local_executor", "scripts", "tests")
        for path in sorted((ROOT / folder).rglob("*.py"))
        for call in _text_calls_without_an_encoding(ast.parse(path.read_bytes(), str(path)))
    ]
    assert not unnamed, "Name the encoding (UTF-8 for the shim's files):\n" + "\n".join(unnamed)


def test_the_check_finds_a_text_call_without_an_encoding() -> None:
    source = """
open(path, "w")
path.open()
path.read_text()
path.write_text(text)
os.fdopen(fd, "w" if write else "rb")
open(path, "rb")
open(path, "w", encoding="utf-8")
os.open(path, flags)
archive.open("manifest.json")
path.open("rb" if binary else "r+b")
"""
    found = [call.lineno for call in _text_calls_without_an_encoding(ast.parse(source))]

    assert sorted(found) == [2, 3, 4, 5, 6]
