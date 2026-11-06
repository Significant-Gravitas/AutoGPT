"""Take out of the bundle what the app never loads.

Every file costs the Windows installer two writes and a virus scan, and an
update a rename, so the count matters as much as the bytes. Each function
removes one kind of thing and returns how many files went; build_runtime.py
(step_prune) calls them and prints the tally. Nothing here guesses: a rule
that depends on what other code does reads that code (the imports of a test
directory, the Google APIs the backend names, RabbitMQ's own dependency
lists), and tests/test_bundle_size.py holds each rule against the tree it was
written for. What the rules cannot see, the build's gates do
(bundle_gate.py), and then the smoke test.
"""

from __future__ import annotations

import ast
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

# Typing stubs and markers, and sources and link libraries of compiled
# extensions: for a type checker and a compiler, neither of which is here.
BUILD_ONLY_SUFFIXES = frozenset(
    {
        ".pyi",
        ".h",
        ".c",
        ".cc",
        ".cpp",
        ".pyx",
        ".pxd",
        ".pxi",
        ".f",
        ".f90",
        ".lib",
        ".a",
    }
)
BUILD_ONLY_NAMES = frozenset({"py.typed"})
TEST_DIRECTORIES = frozenset({"tests", "test"})
# Parts of the standard library that are a desktop GUI, an editor, a demo or
# an installer of pip.
STDLIB_UNUSED = ("tkinter", "idlelib", "turtledemo", "ensurepip")
# Kept in python/Scripts (python/bin elsewhere) besides the interpreter: the
# Prisma CLI starts its Python generator by this name when the build runs
# `prisma generate` (step_prisma).
SCRIPTS_KEPT = re.compile(r"(python|prisma)", re.IGNORECASE)
IMPORT_LINE = re.compile(
    rb"^[ \t]*(?:from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+([^\n#]+)|import[ \t]+([^\n#]+))",
    re.MULTILINE,
)
GOOGLE_CLIENT = "googleapiclient"
# The package, and the old name the same distribution still installs.
GOOGLE_CLIENT_ITSELF = frozenset({GOOGLE_CLIENT, "apiclient"})
DISCOVERY_DOCUMENTS = Path(GOOGLE_CLIENT) / "discovery_cache" / "documents"
# The applications the broker is started with. The first is the server; the
# second is what its launch script boots ahead of it.
RABBITMQ_ROOTS = ("rabbit", "rabbitmq_prelaunch")
RABBITMQ_CLI_KEPT = "rabbitmqctl"
ERLANG_APPLICATIONS = re.compile(
    r"\{\s*(?:applications|included_applications)\s*,\s*\[([^\]]*)\]", re.DOTALL
)


class SlimError(RuntimeError):
    pass


def remove(paths: Iterable[Path]) -> int:
    """Delete files and whole directories; the number of files that went."""
    gone = 0
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            gone += sum(
                1 for entry in path.rglob("*") if entry.is_file() or entry.is_symlink()
            )
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
            gone += 1
    return gone


# --- Python packages ------------------------------------------------------


def unused_pip(packages: Path, requirements: Iterable[str]) -> list[Path]:
    """The interpreter's seeded installer, unless the runtime lock needs it.

    uv installs the bundle's dependencies from outside this interpreter; its
    sync can leave seeded pip behind. Keep all locked packages, including a
    future explicit or transitive pip requirement, regardless of its marker.
    """
    if any(
        re.match(r"pip(?:\s|[<>=!~;\[]|$)", line, re.IGNORECASE)
        for line in requirements
    ):
        return []
    return [
        path
        for path in [packages / "pip", *sorted(packages.glob("pip-*.dist-info"))]
        if path.exists()
    ]


def build_only_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if (path.suffix in BUILD_ONLY_SUFFIXES or path.name in BUILD_ONLY_NAMES)
        and path.is_file()
    ]


def unused_test_directories(packages: Path) -> list[Path]:
    """The `tests` and `test` directories of third-party packages (pandas
    alone ships 2,200 files of them), except any that code outside it
    imports: a package may keep helpers there that it uses at run time."""
    candidates = [
        path
        for path in packages.rglob("*")
        if path.name in TEST_DIRECTORIES
        and path.is_dir()
        and not _inside_another(path, packages)
    ]
    imported = _imported_modules(packages, candidates)
    return [path for path in candidates if _dotted(path, packages) not in imported]


def _inside_another(path: Path, packages: Path) -> bool:
    return any(
        part in TEST_DIRECTORIES for part in path.relative_to(packages).parts[:-1]
    )


def _dotted(path: Path, packages: Path) -> str:
    return ".".join(path.relative_to(packages).parts)


def _imported_modules(packages: Path, test_directories: list[Path]) -> set[str]:
    """Of the test directories' dotted names, those some module outside every
    test directory imports, or imports something from. Read from the import
    statements' text, without resolving what they find: it may keep a
    directory that could have gone, never the other way round."""
    names = {_dotted(path, packages) for path in test_directories}
    found: set[str] = set()
    for source in packages.rglob("*.py"):
        relative = source.relative_to(packages)
        if any(part in TEST_DIRECTORIES for part in relative.parts[:-1]):
            continue
        text = source.read_bytes()
        if b"test" not in text:
            continue
        package = list(relative.parts[:-1])
        for mentioned in _mentioned(text, package):
            found.update(name for name in names if _within(mentioned, name))
    return found


def _mentioned(text: bytes, package: list[str]) -> Iterable[str]:
    for dots, module, names, plain in IMPORT_LINE.findall(text):
        if plain:
            yield from (
                part.split()[0].decode() for part in plain.split(b",") if part.split()
            )
            continue
        base = package[: len(package) - (len(dots) - 1)] if dots else []
        prefix = ".".join([*base, *([module.decode()] if module else [])])
        yield prefix
        for name in names.replace(b"(", b" ").replace(b")", b" ").split(b","):
            if name.split():
                yield f"{prefix}.{name.split()[0].decode()}".lstrip(".")


def _within(mentioned: str, name: str) -> bool:
    return mentioned == name or mentioned.startswith(name + ".")


def unused_discovery_documents(packages: Path, sources: list[Path]) -> list[Path]:
    """google-api-python-client carries the description of every Google API
    there is: 600 JSON files, 100 MB. `build("<api>", "<version>")` reads
    one. Keep the ones the code in `sources` asks for by name."""
    documents = packages / DISCOVERY_DOCUMENTS
    if not documents.is_dir():
        return []
    wanted = {f"{api}.{version}.json" for api, version in google_apis(sources)}
    missing = sorted(name for name in wanted if not (documents / name).is_file())
    if missing:
        raise SlimError(
            f"the backend builds Google API clients for {', '.join(missing)}, which "
            f"{GOOGLE_CLIENT} has no description of. Read how this version finds one "
            "(googleapiclient/discovery.py, discovery_cache) and update desktop/build/slim.py."
        )
    return [path for path in documents.glob("*.json") if path.name not in wanted]


def google_apis(sources: list[Path]) -> set[tuple[str, str]]:
    """Every (api, version) that `googleapiclient.discovery.build` is called
    with, in the Python files under `sources`. A call whose API cannot be
    read off the page stops the build: its description would be deleted and
    the block would fail the first time it ran."""
    found: set[tuple[str, str]] = set()
    for root in sources:
        for path in root.rglob("*.py"):
            if path.relative_to(root).parts[0] in GOOGLE_CLIENT_ITSELF:
                continue  # its own samples and helpers, which pass the name on
            text = path.read_text(encoding="utf-8", errors="replace")
            if GOOGLE_CLIENT in text:
                found |= _build_calls(ast.parse(text, filename=str(path)), path)
    return found


def _build_calls(tree: ast.AST, path: Path) -> set[tuple[str, str]]:
    names = _names_of_build(tree)
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_build(node.func, names):
            continue
        arguments = {keyword.arg: keyword.value for keyword in node.keywords}
        api = _text(node.args[0] if node.args else arguments.get("serviceName"))
        version = _text(
            node.args[1] if len(node.args) > 1 else arguments.get("version")
        )
        if api is None or version is None:
            raise SlimError(
                f"{path}, line {node.lineno}: a Google API client is built for an API that is "
                "not written out in the call. The bundle keeps only the API descriptions the "
                "code names (desktop/build/slim.py); name this one there."
            )
        found.add((api, version))
    return found


def _names_of_build(tree: ast.AST) -> tuple[set[str], set[str]]:
    """What the module calls `googleapiclient.discovery.build` (functions),
    and what it calls the module it is in (modules)."""
    functions: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == f"{GOOGLE_CLIENT}.discovery"
        ):
            functions |= {
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "build"
            }
        elif isinstance(node, ast.ImportFrom) and node.module == GOOGLE_CLIENT:
            modules |= {
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "discovery"
            }
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == f"{GOOGLE_CLIENT}.discovery":
                    modules.add(alias.asname or alias.name)
    return functions, modules


def _is_build(function: ast.expr, names: tuple[set[str], set[str]]) -> bool:
    functions, modules = names
    if isinstance(function, ast.Name):
        return function.id in functions
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "build"
        and ast.unparse(function.value) in modules
    )


def _text(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def pywin32_extras(packages: Path) -> list[Path]:
    """pywin32's editor (Pythonwin, with the MFC runtime it needs), its
    example scripts and its help file."""
    demos = [path for path in packages.glob("win32*/**/[Dd]emos") if path.is_dir()]
    return [
        path
        for path in [packages / "pythonwin", packages / "PyWin32.chm", *demos]
        if path.exists()
    ]


# --- the interpreter --------------------------------------------------------


def interpreter_extras(python: Path) -> list[Path]:
    """Tcl/Tk and the GUI modules built on it, pip's installer, the console
    scripts of the installed packages (they carry the build machine's path
    to the interpreter, and the runtime starts everything through
    `python -c`), import libraries and man pages."""
    windows = (python / "python.exe").is_file()
    library = (
        python / "Lib" if windows else next((python / "lib").glob("python3.*"), python)
    )
    scripts = python / "Scripts" if windows else python / "bin"
    found = [library / name for name in STDLIB_UNUSED]
    found += [python / name for name in ("tcl", "libs", "share", "include")]
    found += (
        [path for path in (python / "lib").glob("*") if _is_tcl(path)]
        if not windows
        else []
    )
    extensions = python / "DLLs" if windows else library / "lib-dynload"
    found += [path for path in extensions.glob("*") if _is_tcl(path)]
    if scripts.is_dir():
        found += [
            path for path in scripts.iterdir() if not SCRIPTS_KEPT.match(path.name)
        ]
    return [path for path in found if path.exists() or path.is_symlink()]


def _is_tcl(path: Path) -> bool:
    return bool(
        re.match(r"(lib)?(_tkinter|tcl|tk|itcl|thread|tix)\d*[._-]?", path.name.lower())
    )


# --- Erlang, RabbitMQ, PostgreSQL ---------------------------------------------


def erlang_extras(erlang: Path) -> list[Path]:
    """The emulator's documentation, headers and sources, the C interface
    libraries under usr/, debugger symbols and the debug build of the VM."""
    found = [erlang / "doc", erlang / "usr"]
    for emulator in erlang.glob("erts-*"):
        found += [emulator / name for name in ("doc", "include", "src", "man")]
        found += emulator.glob("bin/beam.debug.*")
    found += erlang.rglob("*.pdb")
    return [path for path in found if path.exists()]


def unused_rabbitmq_plugins(rabbitmq: Path) -> list[Path]:
    """RabbitMQ ships every plugin it has (MQTT, STOMP, federation, a web
    console, cloud discovery) as a directory under plugins/, and none is
    enabled here. Kept: the server and what its own .app files say it starts,
    all the way down. Erlang/OTP's applications are not among the plugins."""
    plugins = rabbitmq / "plugins"
    directory = {
        _application(path): path for path in plugins.iterdir() if path.is_dir()
    }
    missing = [name for name in RABBITMQ_ROOTS if name not in directory]
    if missing:
        raise SlimError(
            f"RabbitMQ has no {', '.join(missing)} under plugins/ any more: read how this "
            "version lays out its applications and update desktop/build/slim.py"
        )
    needed: set[str] = set()
    pending = list(RABBITMQ_ROOTS)
    while pending:
        name = pending.pop()
        if name in needed or name not in directory:
            continue
        needed.add(name)
        pending += _applications(directory[name], name)
    return [path for name, path in sorted(directory.items()) if name not in needed]


def _application(plugin: Path) -> str:
    """rabbit_common-4.1.8 -> rabbit_common"""
    return plugin.name.rsplit("-", 1)[0]


def _applications(plugin: Path, name: str) -> list[str]:
    resource = plugin / "ebin" / f"{name}.app"
    if not resource.is_file():
        raise SlimError(f"{resource} is missing: cannot tell what {name} depends on")
    listed = ERLANG_APPLICATIONS.findall(
        resource.read_text(encoding="utf-8", errors="replace")
    )
    return [
        item.strip().strip("'")
        for group in listed
        for item in group.split(",")
        if item.strip()
    ]


def unused_rabbitmq_tools(rabbitmq: Path) -> list[Path]:
    """Each command-line tool is the same 7 MB archive under another name.
    The runtime runs the server and `rabbitmqctl stop`."""
    tools = rabbitmq / "escript"
    kept = tools / RABBITMQ_CLI_KEPT
    if not kept.is_file():
        raise SlimError(f"{kept} is missing: the runtime stops the broker with it")
    return [path for path in tools.iterdir() if path.name != RABBITMQ_CLI_KEPT]


def postgres_extras(postgres: Path) -> list[Path]:
    """What is for compiling against PostgreSQL."""
    found = [
        postgres / "include",
        postgres / "lib" / "pgxs",
        postgres / "lib" / "pkgconfig",
    ]
    found += postgres.rglob("*.a")
    return [path for path in found if path.exists()]
