"""The look for the wheel's cua-driver asks nothing of the import system.

importlib.util.find_spec holds the interpreter's import lock while each of
its finders looks, and a finder stats its folder. With a folder of sys.path
on a mount that stopped answering, the look held that lock for as long as
the mount stalled. The look has a thread of its own, so nothing waited for
the look itself; but the first import of any module, on any other thread,
waits for that lock, and the event loop's thread is one of them. The loop
then stopped: no keepalive, no other session's work, and no caller's bound
on the look could run out.

bundled_binary looks in the folders of sys.path itself now, for
`cua_driver/bin/cua-driver`, which is where the wheel puts it.
"""

from __future__ import annotations

import importlib
import importlib._bootstrap_external as import_machinery
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.computer_use import driver

EXECUTABLE = "cua-driver.exe" if sys.platform == "win32" else "cua-driver"


def _a_wheel(folder: Path) -> Path:
    """What the cua-driver wheel leaves in a folder Python imports from."""
    package = folder / "cua_driver"
    (package / "bin").mkdir(parents=True)
    (package / "__init__.py").write_text("raise ImportError('never imported')\n", encoding="utf-8")
    binary = package / "bin" / EXECUTABLE
    binary.write_bytes(b"")
    return binary


def test_the_wheels_driver_is_found_in_a_folder_python_imports_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _a_wheel(tmp_path / "site-packages")
    (tmp_path / "an-archive.zip").write_bytes(b"")
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(tmp_path / "not-there"),
            str(tmp_path / "an-archive.zip"),
            str(tmp_path / "site-packages"),
        ],
    )

    assert driver.bundled_binary() == binary


def test_no_wheel_no_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "site-packages" / "cua_driver").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(tmp_path / "site-packages")])

    assert driver.bundled_binary() is None


def test_the_path_found_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The current folder, as Python reads an empty entry, and a relative
    one: the driver starts in another folder than the look was made in."""
    in_the_current_folder = _a_wheel(tmp_path)
    in_a_relative_one = _a_wheel(tmp_path / "relative")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "path", [""])
    assert driver.bundled_binary() == in_the_current_folder.absolute()

    monkeypatch.setattr(sys, "path", ["relative"])
    assert driver.bundled_binary() == in_a_relative_one.absolute()


def test_the_look_asks_no_import_finder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each finder is asked under the import lock."""
    binary = _a_wheel(tmp_path / "site-packages")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "site-packages")])
    asked: list[str] = []

    class Finder:
        @staticmethod
        def find_spec(name: str, path: Any = None, target: Any = None) -> None:
            asked.append(name)

    monkeypatch.setattr(sys, "meta_path", [Finder, *sys.meta_path])

    assert driver.bundled_binary() == binary
    assert "cua_driver" not in asked
    assert "cua_driver" not in sys.modules


def test_a_look_held_by_a_folder_does_not_hold_another_threads_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder of sys.path stops answering while the look is in it. A
    module no thread has imported yet, in a folder that does answer, is
    imported by another thread meanwhile, as the event loop's thread
    imports one the first time it is needed."""
    name = "a_module_the_shim_first_needs_now"
    answering = tmp_path / "answering"
    answering.mkdir()
    (answering / f"{name}.py").write_text("LOADED = True\n", encoding="utf-8")
    stalled = tmp_path / "stalled"
    stalled.mkdir()
    monkeypatch.setattr(sys, "path", [str(answering), str(stalled)])
    monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()

    held, answers = threading.Event(), threading.Event()
    real_stat = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        # The folder stops answering for the look, and for the look alone:
        # then nothing but the look can keep the import waiting. A coverage
        # tool that measures the import looks at each folder of sys.path
        # for itself, on the importing thread, and that look isn't what
        # this test is about.
        if threading.current_thread() is look and os.fspath(path).startswith(str(stalled)):
            held.set()
            answers.wait(60)
        return real_stat(path, *args, **kwargs)

    # What pathlib asks, and what the import system's finders ask.
    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(import_machinery, "_path_stat", stat)

    looked: list[Path | None] = []
    imported: list[Any] = []
    look = threading.Thread(target=lambda: looked.append(driver.bundled_binary()), daemon=True)
    other = threading.Thread(
        target=lambda: imported.append(importlib.import_module(name)), daemon=True
    )
    try:
        look.start()
        assert held.wait(30), "the look never reached the folder"
        other.start()
        other.join(10)
        was_held = other.is_alive()
    finally:
        # Before anything here imports: a failed assertion's report does.
        answers.set()
        look.join(30)
        other.join(30)
        sys.modules.pop(name, None)

    assert not was_held, "an import on another thread waited for the look"
    assert imported and imported[0].LOADED is True
    assert looked == [None]
