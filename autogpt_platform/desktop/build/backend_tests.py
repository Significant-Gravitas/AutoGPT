"""Leave the backend's own tests out of the bundle.

The backend keeps its tests beside the code: 850 modules, a fifth of the
package. A few modules that are named like tests are not tests
(backend/blocks/exa/_test.py is a block's helper, backend/util/test.py is
imported by the REST API), so the name alone does not decide. A module is
left out when it is named like a test AND no module that stays imports it,
directly or through other test-named modules.

The block loader matters here (backend/blocks/__init__.py): it imports every
module under blocks/ whose name does not start with `test_`, so the 60
`*_test.py` files there were being imported by every executor at start. They
define no block (checked below); without them the executor imports less.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# pytest's own naming rules plus the backend's `<module>_test.py`. `.+`, not
# `.*`: a module called `_test.py` is somebody's helper.
TEST_NAME = re.compile(r"(.+_test|test_.*|conftest)\.py")
BLOCKS = "blocks"


class BlockInTestError(RuntimeError):
    pass


def prune(package: Path) -> list[Path]:
    """Delete the test modules of `package` (the bundle's backend/backend)
    that nothing else imports. Returns what was deleted."""
    doomed = removable(package)
    defining = [path for path in doomed if _defines_a_block(package, path)]
    if defining:
        listed = ", ".join(str(path.relative_to(package)) for path in defining[:5])
        raise BlockInTestError(
            f"{listed}: named like a test, under blocks/, and defines a block. The block "
            "loader would have registered it; leaving the file out would take a block away. "
            "Keep it (desktop/build/backend_tests.py) or have upstream rename it."
        )
    for path in doomed:
        path.unlink()
        compiled = path.parent / "__pycache__"
        for stale in compiled.glob(f"{path.stem}.*.pyc") if compiled.is_dir() else []:
            stale.unlink()
    _remove_empty_directories(package)
    return doomed


def removable(package: Path) -> list[Path]:
    modules = _modules(package)
    tests = {name for name, path in modules.items() if TEST_NAME.fullmatch(path.name)}
    kept = set(modules) - tests
    pending = list(kept)
    while pending:
        name = pending.pop()
        for imported in _imports(name, modules) & tests - kept:
            kept.add(imported)
            pending.append(imported)
    return sorted(modules[name] for name in tests - kept)


def _modules(package: Path) -> dict[str, Path]:
    """Dotted name -> file, for every module of the package."""
    found = {}
    for path in package.rglob("*.py"):
        parts = list(path.relative_to(package.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


def _imports(name: str, modules: dict[str, Path]) -> set[str]:
    """The modules of the package that `name` may import: every `import`
    and `from ... import` anywhere in it (inside functions and under
    TYPE_CHECKING too), and every string that is a module's name, which is
    how a dynamic import would name one. More than it really imports, never
    less."""
    path = modules[name]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    mentioned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mentioned.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node.level, node.module)
            mentioned.add(base)
            mentioned.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            mentioned.add(node.value)
    return {
        _prefix for dotted in mentioned for _prefix in _prefixes(dotted)
    } & modules.keys()


def _resolve(package: str, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


def _prefixes(dotted: str) -> list[str]:
    """a.b.c -> a, a.b, a.b.c: importing a module imports its packages."""
    parts = dotted.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts) + 1)]


def _defines_a_block(package: Path, path: Path) -> bool:
    """Whether the block loader, which skips only `test_*` names, would
    have found a block class in this file."""
    relative = path.relative_to(package)
    if relative.parts[0] != BLOCKS or path.name.startswith("test_"):
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.ClassDef) and node.name.endswith("Block") and node.bases
        for node in tree.body
    )


def _remove_empty_directories(package: Path) -> None:
    for directory in sorted(
        package.rglob("*"), key=lambda path: len(path.parts), reverse=True
    ):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
