"""Computer use is advertised only when every module its backend imports is
installed, and the computer-use extra installs all of them."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from autogpt_local_executor import platform_info

ROOT = Path(__file__).resolve().parents[1]
COMPUTER_USE = ROOT / "autogpt_local_executor" / "computer_use"
BACKENDS = {
    "darwin": COMPUTER_USE / "backends" / "macos.py",
    "linux": COMPUTER_USE / "backends" / "linux.py",
    "windows": COMPUTER_USE / "backends" / "windows.py",
}
SHARED = [COMPUTER_USE / "display.py"]
SYS_PLATFORM = {"darwin": "darwin", "linux": "linux", "windows": "win32"}
# The distribution that installs each module.
DISTRIBUTIONS = {
    "pyautogui": "pyautogui",
    "PIL": "Pillow",
    "mss": "mss",
    "Quartz": "pyobjc-framework-Quartz",
    "AppKit": "pyobjc-framework-Cocoa",
    "Cocoa": "pyobjc-framework-Cocoa",
    "ApplicationServices": "pyobjc-framework-ApplicationServices",
    "win32api": "pywin32",
    "win32clipboard": "pywin32",
    "win32con": "pywin32",
    "win32gui": "pywin32",
    "win32process": "pywin32",
    "win32ui": "pywin32",
}


def _third_party_imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {
        name
        for name in names
        if name not in sys.stdlib_module_names
        and name not in {"__future__", "autogpt_local_executor"}
    }


@pytest.mark.parametrize("platform_name", sorted(BACKENDS))
def test_the_probe_checks_every_module_the_backend_imports(platform_name: str) -> None:
    imported = _third_party_imports(BACKENDS[platform_name]).union(
        *(_third_party_imports(path) for path in SHARED)
    )

    assert imported <= set(platform_info.computer_use_modules(platform_name))


@pytest.mark.parametrize("platform_name", sorted(BACKENDS))
def test_the_computer_use_extra_installs_every_probed_module(platform_name: str) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = [Requirement(r) for r in pyproject["project"]["optional-dependencies"]["computer-use"]]
    environment = {"sys_platform": SYS_PLATFORM[platform_name]}
    installed = {
        requirement.name.lower()
        for requirement in extra
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }

    missing = [
        module
        for module in platform_info.computer_use_modules(platform_name)
        if DISTRIBUTIONS[module].lower() not in installed
    ]
    assert missing == []


@pytest.mark.parametrize(
    ("platform_name", "missing"),
    [
        ("linux", "mss"),
        ("windows", "mss"),
        ("windows", "win32gui"),
        ("darwin", "Quartz"),
        ("darwin", "ApplicationServices"),
    ],
)
def test_computer_use_is_not_advertised_without_a_module_its_backend_needs(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, missing: str
) -> None:
    """pyautogui alone is not enough: screenshots and display bounds need mss,
    and each OS's window and clipboard operations need its native bindings."""
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.setattr(platform_info, "detect_platform", lambda: platform_name)
    monkeypatch.setattr(platform_info, "_module_available", lambda name: name != missing)

    assert "computer_use" not in platform_info.detect_capabilities(enable_computer_use=True)


@pytest.mark.parametrize("platform_name", sorted(BACKENDS))
def test_computer_use_is_advertised_with_every_module_installed(
    monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.setattr(platform_info, "detect_platform", lambda: platform_name)
    monkeypatch.setattr(platform_info, "_module_available", lambda name: True)

    assert "computer_use" in platform_info.detect_capabilities(enable_computer_use=True)
