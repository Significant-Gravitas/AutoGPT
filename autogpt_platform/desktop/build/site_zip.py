"""Pack the pure-Python packages into a few files.

Two thirds of the bundle's files are third-party Python modules and their
bytecode: 16,000 of each. Python imports from a zip as readily as from a
directory, and a zip is one file to install, scan and replace instead of
thousands. The archives are in `site-zip/`, beside `site/` and after it on
the interpreter's path (build_runtime.py, step_relocate).

What goes in is decided a package at a time, never part of one: Python
cannot find one regular package in two places. A package goes in when
everything under it is Python source, or when its top-level name is on
DATA_READ_THROUGH_THE_LOADER and it has no other file than that list says.
It stays out when it has anything else (a compiled extension, a certificate
bundle, a template), when a .pth file imports it, when it is on ON_DISK, or
when a module of it names its own file (`__file__`, `__path__`): code that
does may open what is beside it by path, which a zip does not have (firecrawl
read its version that way, and reported a wrong one from a zip without
failing). The packages on NAMES_ITS_FILES do and go in all the same, each use
having been read. A directory without an `__init__.py`
and with nothing but directories in it is a namespace (`google`,
`opentelemetry`), which Python does assemble from several places: the
packages inside it are decided one by one. Top-level modules that are single
files, and every *.dist-info (importlib.metadata reads those as files), stay
where they are.

In a zip each module is stored with its bytecode beside it (`mod.py` and
`mod.pyc`, which is where zipimport looks), uncompressed: the installer
compresses the whole bundle anyway, and an uncompressed entry is read
without inflating it. The bytecode is the unchecked-hash kind, as for loose
files (step_compile), so it is trusted without a look at the source.
`without_bytecode` holds an archive to that: a module that is in one without
its bytecode imports all the same, compiled again by every process of every
start, and nothing else would show it.

What a module from a zip knows about itself differs from a file's in two
ways. Its `__file__` ends in `.pyc` (`.../site-zip/15.zip/openai/_client.pyc`):
zipimport names the entry it loaded. And its code is compiled under a name
that is the same on every build machine (`site-zip/openai/_client.py`), which
zipimport does not replace with where the archive is, as the import of a
.pyc file does. The second is put right as the interpreter starts
(build_runtime.RELOCATION): code loaded from an archive is renamed to its
source's place in it, so tracebacks, and packages that tell their own frames
from a caller's by file name (neo4j, e2b), see what they would see of files.

There are SHARDS archives and not one because of updates. An update
downloads the parts of the new installer that differ from the old one, and
the installer compresses each file of the bundle by itself: when a package
changes size, everything behind it in its archive moves, and the rest of
that archive compresses to other bytes. In one archive of everything that
was 16 MB of download for 16 bytes more in the middle (measured on the
Windows installer, 2026-10-03, the zipped packages being 28 MB of it); with
the packages dealt out over 32 archives by their names, a package that
changes takes its own archive with it and no other: 1 MB for the same 16
bytes in the archive that has openai, 4 MB in the largest (yt-dlp's). Which
archive a package is in depends on its name alone, so it stays there from
one build to the next. More archives would make an update smaller still and
every start slower: each is one more place the interpreter looks.

The standard library is not zipped, on any system. On Windows a
`python313.zip` beside `python313.dll` looks like the way to do it (it is
the layout of python.org's embeddable distribution), and it makes the
interpreter find its home by that archive instead of by `Lib/os.py`. Found
that way, it also takes the search path of any Python 3.13 installed on the
machine from the registry (CPython's getpath.py reads
the key `PythonCore/3.13/PythonPath` under `Software/Python` unless the standard library
was found as files, or a `._pth` file or `-E` turns the registry and every
PYTHON* variable off). That installation's modules and extension modules
then load in place of the bundle's and of what was pruned from it. A build
on GitHub's Windows machines, which have one, showed it: `tkinter`, which
the bundle does not have, imported. `unpack_stdlib` puts back a bundle that
was built that way, and bundle_gate.py checks where the interpreter looks.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import warnings
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeGuard

ARCHIVES = "site-zip"
SHARDS = 32
PACKAGES = "site"
SOURCE = ".py"
COMPILED = ".pyc"
INIT = "__init__.py"

# Top-level packages that are not only Python source and still go into the
# zip, each for a reason that was looked up: its other files are read through
# the import system (importlib.resources, pkgutil.get_data), which serves
# them from a zip, or are read by nothing that runs here.
#
# With each, the files that were looked at (fnmatch patterns, relative to the
# package; `*` crosses directories). A version of the package that has
# another one stops the build: nobody has looked up how that one is read.
DATA_READ_THROUGH_THE_LOADER: dict[str, tuple[str, tuple[str, ...]]] = {
    "tzdata": (
        "the time zone files, which zoneinfo opens with importlib.resources",
        ("zoneinfo/*", "zones"),
    ),
    "yt_dlp": (
        "three .js files read with pkgutil.get_data; upstream ships itself as a zip",
        ("extractor/youtube/jsc/_builtin/vendor/*.js",),
    ),
    "urllib3": (
        "a JavaScript worker for Pyodide, read there with importlib.resources",
        ("contrib/emscripten/emscripten_fetch_worker.js",),
    ),
    "tqdm": (
        "a shell completion and a man page, which only `tqdm --manpath` reads",
        ("completion.sh", "tqdm.1"),
    ),
    "bleach": (
        "notes on the html5lib it vendors and that copy's metadata, read by nothing",
        (
            "_vendor/html5lib-*.dist-info/*",
            "_vendor/README.rst",
            "_vendor/parse.py.SHA256SUM",
            "_vendor/vendor.txt",
            "_vendor/vendor_install.sh",
        ),
    ),
    "openai": ("an empty .keep file", ("lib/.keep",)),
    "anthropic": (
        "an empty .keep file and a Markdown note",
        ("lib/.keep", "lib/foundry.md"),
    ),
    "groq": ("an empty .keep file", ("lib/.keep",)),
    "langfuse": ("two Markdown notes", ("api/README.md", "api/reference.md")),
    "langsmith": ("two Markdown notes", ("cli/README.md", "sandbox/README.md")),
    "markdown_it": ("a YAML note about the port, read by nothing", ("port.yaml",)),
}
# A file of these kinds is loaded by its path, whatever the list above says.
NATIVE_SUFFIXES = (".pyd", ".so", ".dylib", ".dll")
# Packages with a module that names its own file and that go into the zip
# all the same, by the modules that do (paths under site/) and what each was
# seen to do with it (read 2026-10-03). A use in any other module of one of
# these packages stops the build: it is one nobody has read. A package that is
# not here and names its files stays on disk without a word.
NAMES_ITS_FILES: dict[str, dict[str, str]] = {
    "anyio": {
        "anyio/_lazyimport.py": "a label for compile(); the source is asked of the loader",
    },
    "e2b": {
        "e2b/template/utils.py": "tells its frames from a caller's by directory (RELOCATION)",
    },
    "imageio": {
        "imageio/core/findlib.py": "LOCALDIR, which nothing reads",
        "imageio/core/util.py": "THIS_DIR, which nothing reads",
        "imageio/plugins/pillow_info.py": "rewrites itself from Pillow, for imageio's maintainers",
        "imageio/testing.py": "helpers of imageio's own tests",
    },
    "neo4j": {
        "neo4j/_async/work/result.py": "tells its frames from a caller's by directory (RELOCATION)",
        "neo4j/_sync/work/result.py": "tells its frames from a caller's by directory (RELOCATION)",
    },
    "posthog": {
        "posthog/exception_utils.py": "another module's file, to shorten a frame's path in a report",
    },
    "pydantic": {
        "pydantic/version.py": "version_info() looks for a git checkout around the package",
        "pydantic/v1/version.py": "version_info() prints where the package is",
    },
    "pygments": {
        "pygments/sphinxext.py": "a Sphinx extension: other modules' files, as what a page depends on",
    },
    "rich": {
        "rich/pretty.py": "the standard library's files, to know a default repr by its code",
        "rich/traceback.py": "the directory of a module a caller wants left out of a traceback",
    },
    "sentry_sdk": {
        "sentry_sdk/utils.py": "another module's file, to shorten a frame's path in a report",
    },
    "yt_dlp": {
        "yt_dlp/__pyinstaller/__init__.py": "tells PyInstaller where its hooks are",
        "yt_dlp/plugins.py": "leaves its own directory out of where it looks for plugins",
        "yt_dlp/update.py": "how it was installed, with a branch for a zip: upstream ships as one",
    },
}
OWN_FILE_NAMES = ("__file__", "__path__")
NAMES_OWN = "names its own files"
# Pure Python, and kept as files all the same.
ON_DISK = {
    "setuptools": "pkg_resources and distutils-precedence.pth find it by its files",
    "pkg_resources": "walks its own directory; aioclamd imports it as the backend starts",
    "_distutils_hack": "imported by distutils-precedence.pth before the zips are on the path",
    "prisma": "the build writes the generated client into it, and it runs its engine by path",
    "claude_agent_sdk": "the runtime reads its version files, and it carries the Claude Code CLI",
    "pytest": "rewrites the modules it imports, from their files",
    "_pytest": "rewrites the modules it imports, from their files",
    "autogpt_libs": "the platform's own code, like backend/: logs and tracebacks name its files",
    "firecrawl": "reads its version out of its own __init__.py, by path, for every request it sends",
}
# Distributions of which nothing goes into the zip, whatever directories they
# install (their RECORD says which). pywin32 spreads over half a dozen, finds
# its extension modules by widening the packages' own search paths, and
# registers COM servers by file name.
ON_DISK_DISTRIBUTIONS = {
    "pywin32": "its packages find their extension modules beside themselves, by path",
}
# What a .pth file does when the interpreter starts: `import x` lines.
PTH_IMPORT = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
STDLIB_ARCHIVE = "python{major}{minor}.zip"
# Of the standard library's directory, what stays a directory on Windows.
# One fixed time for every entry: an archive is the same bytes whenever it
# is built from the same files, which is what a differential update compares.
EPOCH = (2020, 1, 1, 0, 0, 0)
# The flags word of bytecode that is trusted without a look at its source.
UNCHECKED_HASH = 0b01


class ZipError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plan:
    # Directories under site/, as posix paths: `openai`, `google/auth`.
    zipped: tuple[str, ...]
    kept: dict[str, str]  # directory -> why it stays on disk

    @property
    def modules(self) -> list[str]:
        """The zipped packages by the name they are imported under."""
        return [unit.replace("/", ".") for unit in self.zipped]


def shard(unit: str) -> str:
    """The archive a package goes into: by its name and nothing else."""
    return f"{int(hashlib.sha256(unit.encode()).hexdigest(), 16) % SHARDS:02d}.zip"


def plan(packages: Path) -> Plan:
    """Which packages under `packages` go into the zip."""
    imported_by_pth = _pth_imports(packages)
    owned = _installed_by(packages, ON_DISK_DISTRIBUTIONS)
    zipped, kept = [], {}
    for entry in sorted(packages.iterdir()):
        name = entry.name
        if (
            not entry.is_dir()
            or name == "__pycache__"
            or name.endswith((".dist-info", ".egg-info"))
        ):
            continue
        if name in ON_DISK:
            kept[name] = ON_DISK[name]
        elif name in owned:
            kept[name] = ON_DISK_DISTRIBUTIONS[owned[name]]
        elif name in imported_by_pth:
            kept[name] = "a .pth file imports it"
        elif name in DATA_READ_THROUGH_THE_LOADER:
            _check_listed_data(entry)
            _check_read(name, _naming_own_files(entry, packages))
            zipped.append(name)
        else:
            for unit in _units(entry):
                relative = unit.relative_to(packages).as_posix()
                other = next(_other_files(unit), None)
                naming = [] if other else _naming_own_files(unit, packages)
                if other:
                    kept[relative] = (
                        f"has files that are not Python source ({other.name})"
                    )
                elif naming and relative not in NAMES_ITS_FILES:
                    kept[relative] = f"{NAMES_OWN} ({naming[0]})"
                else:
                    _check_read(relative, naming)
                    zipped.append(relative)
    return Plan(tuple(zipped), kept)


def _check_listed_data(package: Path) -> None:
    """A package on DATA_READ_THROUGH_THE_LOADER has the files that were
    looked at when it was put there, and no other."""
    _, patterns = DATA_READ_THROUGH_THE_LOADER[package.name]
    others = sorted(
        path.relative_to(package).as_posix() for path in _other_files(package)
    )
    unknown = [
        name
        for name in others
        if name.endswith(NATIVE_SUFFIXES)
        or not any(fnmatch.fnmatchcase(name, p) for p in patterns)
    ]
    if unknown:
        raise ZipError(
            f"{package.name} has files that DATA_READ_THROUGH_THE_LOADER in "
            f"desktop/build/site_zip.py does not list for it: {', '.join(unknown[:5])}"
            f"{' and more' if len(unknown) > 5 else ''}. It is zipped on the word that its "
            "data is read through the import system or not at all, and nobody has looked at "
            "these. Find what in the package reads them: when it is importlib.resources or "
            "pkgutil.get_data, or nothing, add them to its entry; when it is a path (open, "
            "Path(__file__)), take the package off the list, and it stays on disk."
        )


def _check_read(unit: str, naming: list[str]) -> None:
    unread = [
        module for module in naming if module not in NAMES_ITS_FILES.get(unit, {})
    ]
    if unread:
        raise ZipError(
            f"{', '.join(unread[:5])} of {unit} names its own file (__file__ or __path__), "
            "and NAMES_ITS_FILES in desktop/build/site_zip.py does not have that module. The "
            "package is zipped because every such use in it was read and found not to open "
            "a file beside the module by path; this one is new. Read it: add the module "
            "with what it does when that holds, or take the package off NAMES_ITS_FILES "
            "(and DATA_READ_THROUGH_THE_LOADER), and it stays on disk."
        )


def _naming_own_files(package: Path, packages: Path) -> list[str]:
    """The modules under `package` that read `__file__` or `__path__`, as
    paths under `packages`."""
    found = []
    for path in sorted(package.rglob("*" + SOURCE)):
        if path.name == "__main__.py":  # a program; nothing imports it
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(name in text for name in OWN_FILE_NAMES) and names_own_file(text):
            found.append(path.relative_to(packages).as_posix())
    return found


def names_own_file(source: str) -> bool:
    """Whether a module reads `__file__` or `__path__`, its own or another
    module's, anywhere but under `if __name__ == "__main__":`. One that
    cannot be parsed is taken to."""
    try:
        pending: list[ast.AST] = [ast.parse(source)]
    except (SyntaxError, ValueError):
        return True
    while pending:
        node = pending.pop()
        if _is_main_guard(node):
            pending.extend(node.orelse)
            continue
        named = node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)
        if named in OWN_FILE_NAMES and isinstance(getattr(node, "ctx", None), ast.Load):
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def _is_main_guard(node: ast.AST) -> TypeGuard[ast.If]:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    sides = [node.test.left, *node.test.comparators]
    names = [side.id for side in sides if isinstance(side, ast.Name)]
    values = [side.value for side in sides if isinstance(side, ast.Constant)]
    return len(sides) == 2 and names == ["__name__"] and values == ["__main__"]


def _units(directory: Path) -> Iterator[Path]:
    """The directories that are decided as a whole: a regular package, or a
    directory with files of its own. A namespace with nothing but packages
    in it is looked into instead."""
    children = [
        child for child in sorted(directory.iterdir()) if child.name != "__pycache__"
    ]
    if (directory / INIT).is_file() or any(child.is_file() for child in children):
        yield directory
        return
    for child in children:
        yield from _units(child)


def _other_files(package: Path) -> Iterator[Path]:
    for path in package.rglob("*"):
        if (
            path.is_file()
            and path.suffix != SOURCE
            and path.parent.name != "__pycache__"
        ):
            yield path


def _installed_by(packages: Path, distributions: dict[str, str]) -> dict[str, str]:
    """Top-level name -> distribution, for everything the RECORD of each of
    `distributions` lists."""
    found: dict[str, str] = {}
    for record in packages.glob("*.dist-info/RECORD"):
        name = record.parent.name.removesuffix(".dist-info").rsplit("-", 1)[0]
        distribution = re.sub(r"[-_.]+", "-", name).lower()
        if distribution in distributions:
            lines = record.read_text(encoding="utf-8", errors="replace").splitlines()
            # RECORD paths are written with forward slashes on every system.
            found.update(
                {line.split("/")[0]: distribution for line in lines if "/" in line}
            )
    return found


def _pth_imports(packages: Path) -> set[str]:
    found: set[str] = set()
    for pth in packages.glob("*.pth"):
        text = pth.read_text(encoding="utf-8", errors="replace")
        found |= {name.split(".")[0] for name in PTH_IMPORT.findall(text)}
        # A line that is not an import is a directory to add to the path.
        found |= {
            re.split(r"[\\/]", line.strip())[0]
            for line in text.splitlines()
            if line.strip()
        }
    return found


def pack(root: Path, python: Path) -> Plan:
    """Move the packages `plan` names from <root>/site into the archives of
    <root>/site-zip, with bytecode compiled by the bundle's own interpreter.
    Run again on a bundle that is packed already, it finds nothing to move."""
    packages = root / PACKAGES
    archives = root / ARCHIVES
    _check_not_split(packages, archives)
    chosen = plan(packages)
    if not chosen.zipped:
        return chosen
    directories = [packages / unit for unit in chosen.zipped]
    _compile(python, packages, directories, ARCHIVES)
    archives.mkdir(exist_ok=True)
    for name in sorted({shard(unit) for unit in chosen.zipped}):
        members = [packages / unit for unit in chosen.zipped if shard(unit) == name]
        _write(archives / name, packages, members)
    for directory in directories:
        shutil.rmtree(directory)
        _remove_emptied(directory.parent, packages)
    _check_not_split(packages, archives)
    check_bytecode(root)
    return chosen


def _remove_emptied(directory: Path, stop: Path) -> None:
    """A namespace whose packages all went into the zip."""
    while directory != stop and not any(directory.iterdir()):
        directory.rmdir()
        directory = directory.parent


def unpack(root: Path) -> None:
    """Put the zipped packages back as files (without their bytecode), for
    a build step that needs to see everything that is installed."""
    archives = root / ARCHIVES
    for archive in sorted(archives.glob("*.zip")) if archives.is_dir() else []:
        with zipfile.ZipFile(archive) as bundle:
            members = [
                name for name in bundle.namelist() if not name.endswith(COMPILED)
            ]
            bundle.extractall(root / PACKAGES, members)
    shutil.rmtree(archives, ignore_errors=True)


def packed(root: Path) -> list[str]:
    """The packages in the archives, as `plan` named them: the outermost
    directories that have a file of their own."""
    archives = root / ARCHIVES
    holders: set[str] = set()
    for archive in sorted(archives.glob("*.zip")) if archives.is_dir() else []:
        holders |= {str(PurePosixPath(name).parent) for name in _files(archive)}
    return sorted(
        holder
        for holder in holders
        if not any(str(parent) in holders for parent in PurePosixPath(holder).parents)
    )


def _files(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as bundle:
        return [name for name in bundle.namelist() if not name.endswith("/")]


def _compile(python: Path, base: Path, directories: list[Path], shown_as: str) -> None:
    """`mod.pyc` beside every `mod.py`. Bytecode in a zip is not told where
    it was loaded from, so the name it is compiled under is one that is the
    same on every build machine, `site-zip/<package>/<module>.py`: the
    archives are then the same bytes wherever they are built. The interpreter
    renames the code as it loads it (build_runtime.RELOCATION)."""
    for directory in directories:
        for cache in list(directory.rglob("__pycache__")):
            shutil.rmtree(cache)
    command = [
        str(python),
        "-B",
        "-m",
        "compileall",
        "-q",
        "-f",
        "-b",
        "-j",
        "0",
        "--invalidation-mode",
        "unchecked-hash",
        "-s",
        str(base),
        "-p",
        shown_as,
    ]
    # Exit code 1 is "some file did not compile": a few packages carry
    # Python 2 sources or templates named .py. Those import, or fail to, from
    # source exactly as they did as files. In batches: a command line has a
    # length limit, and there are hundreds of directories.
    for start in range(0, len(directories), 50):
        batch = [str(directory) for directory in directories[start : start + 50]]
        subprocess.run([*command, *batch], check=False, stdin=subprocess.DEVNULL)


def check_bytecode(root: Path, *more: Path) -> None:
    """Every archive of <root>/site-zip, and of `more`, has its modules'
    bytecode. Asked when they are written and again as the bundle is sealed:
    nothing else notices bytecode that is missing."""
    archives = root / ARCHIVES
    found = [
        f"{archive.name}: {name}"
        for archive in [
            *(sorted(archives.glob("*.zip")) if archives.is_dir() else []),
            *more,
        ]
        for name in without_bytecode(archive)
    ]
    if found:
        raise ZipError(
            f"{len(found)} modules are zipped without bytecode a start can use ({', '.join(found[:5])}"
            f"{' and more' if len(found) > 5 else ''}). Each would be compiled by every "
            "process that imports it, on every start. The compile inside the zip step did "
            "not run or did not finish (site_zip._compile): run the zip step again on an "
            "unzipped bundle (deps, relocate, compile, zip) and read what compileall prints."
        )


def without_bytecode(archive: Path) -> list[str]:
    """The modules in `archive` that have no unchecked-hash bytecode beside
    them, though they compile. A file that is not Python 3 (a template, a
    Python 2 source) has none, and imports from a zip, or fails to, as it
    did as a file."""
    found = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for name in sorted(name for name in names if name.endswith(SOURCE)):
            if name + "c" in names:
                with bundle.open(name + "c") as stream:
                    flags = int.from_bytes(stream.read(8)[4:], "little")
                if flags != UNCHECKED_HASH:
                    found.append(f"{name}c (its flags are {flags}, not unchecked-hash)")
            elif _compiles(bundle.read(name)):
                found.append(name)
    return found


def _compiles(source: bytes) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            compile(source, "<zipped>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return False
    return True


def _write(archive: Path, base: Path, directories: list[Path]) -> None:
    with zipfile.ZipFile(
        archive, "a" if archive.is_file() else "w", zipfile.ZIP_STORED
    ) as bundle:
        named = set(bundle.namelist())
        for directory in directories:
            # Named explicitly, parents included: a package with no
            # __init__.py is found by its directory entry.
            parents = [
                *reversed(directory.relative_to(base).parents[:-1]),
                directory.relative_to(base),
            ]
            for parent in parents:
                _add_directory(bundle, parent.as_posix(), named)
            for path in sorted(directory.rglob("*")):
                name = path.relative_to(base).as_posix()
                if path.is_dir():
                    _add_directory(bundle, name, named)
                else:
                    bundle.writestr(zipfile.ZipInfo(name, EPOCH), path.read_bytes())


def _add_directory(bundle: zipfile.ZipFile, name: str, named: set[str]) -> None:
    if name + "/" not in named:
        named.add(name + "/")
        bundle.writestr(zipfile.ZipInfo(name + "/", EPOCH), b"")


def _check_not_split(packages: Path, archives: Path) -> None:
    """No package may be zipped and on disk at once, or in two archives:
    Python would import one of the two and never see the other. A namespace
    may be."""
    seen: dict[str, str] = {}
    twice: list[str] = []
    for archive in sorted(archives.glob("*.zip")) if archives.is_dir() else []:
        regular = {
            str(PurePosixPath(name).parent)
            for name in _files(archive)
            if name.endswith("/" + INIT)
        }
        twice += [
            f"{name} ({seen[name]}, {archive.name})"
            for name in sorted(regular & seen.keys())
        ]
        seen.update(dict.fromkeys(regular, archive.name))
    both = sorted(
        name for name in {*seen, *packed(archives.parent)} if (packages / name).exists()
    )
    if both or twice:
        raise ZipError(
            f"{[*both, *twice][:5]} are in {archives.name}/ and in {packages.name}/ at once, or "
            "in two archives. Python would import one of the two and never see the other. "
            "Run the deps step and what follows it (relocate, compile, zip) again."
        )


# --- the standard library, on Windows ---------------------------------------


def unpack_stdlib(python_home: Path, version: tuple[int, int]) -> bool:
    """Put a zipped standard library back as files in Lib/ (see the module
    docstring for why it is not zipped). True when there was one: its
    bytecode is not put back, so the caller compiles Lib/ again."""
    archive = python_home / STDLIB_ARCHIVE.format(major=version[0], minor=version[1])
    if not archive.is_file():
        return False
    library = python_home / "Lib"
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if not name.endswith(COMPILED)]
        bundle.extractall(library, members)
    archive.unlink()
    return True


def zipped_stdlib(python_home: Path) -> list[Path]:
    """The archives the interpreter would take for its standard library."""
    return sorted(python_home.glob("python3*.zip"))


def longest_path(root: Path) -> tuple[int, str]:
    longest = max(
        (str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()),
        key=len,
        default="",
    )
    return len(longest), longest


def count(root: Path) -> tuple[int, int]:
    """(files, bytes) of everything under `root`."""
    files = size = 0
    for directory, _, names in os.walk(root):
        for name in names:
            files += 1
            size += os.lstat(os.path.join(directory, name)).st_size
    return files, size
