"""What pruning and zipping must not have broken, asked of the bundle itself.

    <runtime>/python/python -B bundle_gate.py <runtime> [--imports FILE [--record]]

Run with the bundle's own interpreter, like the smoke test, and before it:
this takes two minutes and starts nothing.

  1. Every module of the backend imports. A module that does not is either
     on NOT_IMPORTABLE, with the reason, or a file the pruning should not
     have taken (build/backend_tests.py, build/slim.py, build/lock_export.py).
  2. `backend.blocks.load_all_blocks()` loads every block: the loader
     imports every file under blocks/, so this is what any service does as
     it starts.
  3. What packages keep as data still reaches them: a time zone through
     zoneinfo and through pytz, the certificate bundle, the version of an
     installed distribution, and aioclamd, which needs pkg_resources. And
     code loaded from a zip names the file it came from, as code loaded from
     a file does (build_runtime.RELOCATION).
  4. Every module of every zipped package (site-zip/) imports from its zip
     exactly as it did from files. `--imports FILE --record` writes down, before
     the packages are zipped, which modules import and which fail and how;
     `--imports FILE` afterwards compares. A module that failed before and
     fails the same way now is not this build's doing (an optional
     dependency that is not installed, a module for another platform). Each
     package is imported in the environment a service has: one that needs
     it (moviepy asks where ffmpeg is) would otherwise fail before and
     after, and none of its modules would have been tried from the zip. A
     zipped package that does not import at all is a finding for that
     reason, unless it is on MAY_NOT_IMPORT.

Exit code 1 when anything is wrong, with one line for each thing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

BUILD = Path(__file__).resolve().parent
DESKTOP = BUILD.parent
FINDING = "GATE: "
CONTRACT_DIRECTORY = DESKTOP / "runtime" / "tests"
ARCHIVES = "site-zip"
PACKAGES = "site"

# Backend modules that cannot be imported in the bundle, and why that is
# right. Anything else that fails is a finding.
NOT_IMPORTABLE: dict[str, str] = {
    "backend.check_db": "a developer's script; it needs faker, which is a development dependency",
}
# Zipped packages that do not import in the bundle, and why that is right.
# None of the modules under such a package is compared, so each is one whose
# modules were looked at by hand.
MAY_NOT_IMPORT: dict[str, str] = {}
# Seconds one package's modules get to import, in a process of their own.
PACKAGE_SECONDS = 600


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument(
        "--imports", type=Path, help="the record of what imported before zipping"
    )
    parser.add_argument(
        "--record", action="store_true", help="write the record instead of comparing"
    )
    # What this script runs itself as, in interpreters of their own.
    parser.add_argument("--package", help=argparse.SUPPRESS)
    parser.add_argument("--in-the-backend", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if args.package:
        as_a_service(runtime, Path.cwd())
        # Not this directory's modules: a package may share a name with one.
        ours = (BUILD, CONTRACT_DIRECTORY)
        sys.path[:] = [
            entry
            for entry in sys.path
            if not entry or Path(entry).resolve() not in ours
        ]
        print(json.dumps(import_package(args.package)))
        sys.stdout.flush()
        os._exit(0)  # as backend_checks: nothing a package started holds this up
    if args.in_the_backend:
        backend_checks(runtime)
    if args.record:
        if not args.imports:
            parser.error("--record needs --imports")
        outcomes = import_packages(runtime, to_be_zipped(runtime))
        args.imports.write_text(
            json.dumps(outcomes, indent=0, sort_keys=True), encoding="utf-8"
        )
        failing = sum(outcome != "ok" for outcome in outcomes.values())
        print(
            f"  recorded {len(outcomes)} modules ({failing} do not import as files either)"
        )
        return 0
    problems = (
        looks_outside(runtime)
        + in_the_backend(runtime)
        + zipped_imports(runtime, args.imports)
    )
    for problem in problems:
        print(f"{FINDING}{problem}")
    print(
        "the bundle passed its gates" if not problems else f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


# --- 0: the interpreter looks for modules in the bundle and nowhere else ------


def looks_outside(runtime: Path) -> list[str]:
    """The search path of the bundle's interpreter, started as the app starts
    it and with no PYTHON* variable set: every entry must be inside the
    bundle. One that is not means another Python's modules can load in place
    of the bundle's. On Windows the registry is how that happens (an
    installed Python of the same version, when the standard library is not
    found as files: site_zip.py), so this only has teeth on a machine that
    has such a Python, as GitHub's have."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("PYTHON")
    }
    answer = subprocess.run(
        [sys.executable, "-B", "-c", "import json, sys; print(json.dumps(sys.path))"],
        capture_output=True,
        text=True,
        cwd=runtime,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    if answer.returncode != 0:
        return [f"the interpreter did not say where it looks: {answer.stderr[-500:]}"]
    outside = entries_outside(json.loads(answer.stdout), runtime)
    print(
        f"  the interpreter looks in {len(json.loads(answer.stdout))} places, {len(outside)} outside the bundle"
    )
    return [
        f"the interpreter also looks for modules in {entry}, which is not in the bundle: another "
        "Python's modules would load in place of the bundle's (build/site_zip.py says how this "
        "happens on Windows)"
        for entry in outside
    ]


def entries_outside(search_path: list[str], runtime: Path) -> list[str]:
    """The entries of a module search path that are not inside `runtime`.
    The empty entry is the working directory, which is the bundle's."""
    root = runtime.resolve()
    return [
        entry
        for entry in search_path
        if entry and not Path(entry).resolve().is_relative_to(root)
    ]


# --- 1 to 3: in one interpreter, set up as a service's is --------------------


def in_the_backend(runtime: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-B", __file__, str(runtime), "--in-the-backend"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )
    lines = result.stdout.splitlines()  # among the backend's own logging
    said = [line.removeprefix(FINDING) for line in lines if line.startswith(FINDING)]
    print("\n".join(line for line in lines if line.startswith("  ")))
    if result.returncode not in (0, 1) or (result.returncode == 1) != bool(said):
        tail = (result.stdout + result.stderr)[-2000:]
        return [
            f"the checks inside the backend ended with code {result.returncode}: {tail}"
        ]
    return said


def backend_checks(runtime: Path) -> NoReturn:
    """Inside an interpreter set up as a service's is."""
    scratch = tempfile.mkdtemp(prefix="autogpt-gate-")
    as_a_service(runtime, Path(scratch))
    problems = unimportable_backend_modules()
    problems += blocks_that_do_not_load()
    problems += data_that_is_not_reached()
    problems += code_named_for_another_file(zipped(runtime))
    for problem in problems:
        print(f"{FINDING}{problem}")
    sys.stdout.flush()
    shutil.rmtree(scratch, ignore_errors=True)
    # Without finalisation: the backend's telemetry threads can hold it up.
    os._exit(1 if problems else 0)


def as_a_service(runtime: Path, scratch: Path) -> None:
    """The working directory and the environment the runtime gives a service
    (backend_contract.py), with `scratch` for a data directory."""
    sys.path.insert(0, str(CONTRACT_DIRECTORY))
    import backend_contract

    backend_contract.as_a_service_sees_it(runtime, scratch)


def unimportable_backend_modules() -> list[str]:
    outcomes = import_all("backend")
    failed = {name: describe(exc) for name, exc in outcomes.items() if exc is not None}
    print(f"  {len(outcomes)} backend modules, {len(failed)} do not import")
    unexpected = sorted(set(failed) - set(NOT_IMPORTABLE))
    stale = sorted(
        name for name in NOT_IMPORTABLE if name in outcomes and name not in failed
    )
    return [
        *[
            f"backend module {name} does not import: {failed[name]}"
            for name in unexpected
        ],
        *[
            f"{name} imports now: take it off NOT_IMPORTABLE in build/bundle_gate.py"
            for name in stale
        ],
    ]


def blocks_that_do_not_load() -> list[str]:
    try:
        from backend.blocks import load_all_blocks

        blocks = load_all_blocks()
    except BaseException as exc:
        return [f"backend.blocks.load_all_blocks() failed: {describe(exc)}"]
    print(f"  {len(blocks)} blocks load")
    return [] if blocks else ["backend.blocks.load_all_blocks() found no block at all"]


def data_that_is_not_reached() -> list[str]:
    checks = {
        "zoneinfo.ZoneInfo('Europe/Paris')": lambda: __import__("zoneinfo").ZoneInfo(
            "Europe/Paris"
        ),
        "pytz.timezone('Asia/Tokyo')": lambda: __import__("pytz").timezone(
            "Asia/Tokyo"
        ),
        "the certificate bundle certifi.where() names": lambda: Path(
            __import__("certifi").where()
        ).read_bytes(),
        "importlib.metadata.version('openai')": lambda: importlib.import_module(
            "importlib.metadata"
        ).version("openai"),
        "import aioclamd (which imports pkg_resources)": lambda: importlib.import_module(
            "aioclamd"
        ),
    }
    problems = []
    for what, check in checks.items():
        try:
            check()
        except BaseException as exc:
            problems.append(f"{what} failed: {describe(exc)}")
    print(f"  {len(checks) - len(problems)} of {len(checks)} data checks pass")
    return problems


def code_named_for_another_file(packages: list[str]) -> list[str]:
    """Code compiled into an archive carries a name that is the same on every
    build machine; the interpreter renames it to its source's place in the
    archive as it loads it. Without that, a traceback names a file that is
    nowhere, and a package that tells its frames from a caller's by file
    name (site_zip.NAMES_ITS_FILES) takes its own for the caller's."""
    wrong = []
    for package in packages:
        spec = importlib.util.find_spec(package)
        get_code = getattr(spec.loader, "get_code", None) if spec else None
        if spec is None or spec.origin is None or get_code is None:
            continue  # a namespace: no code of its own
        code = get_code(package)
        if code is not None and code.co_filename != spec.origin.removesuffix("c"):
            wrong.append(f"{package} ({code.co_filename})")
    print(
        f"  {len(packages) - len(wrong)} of {len(packages)} zipped packages' code names its file"
    )
    if not wrong:
        return []
    return [
        f"code loaded from {ARCHIVES}/ does not name the file it came from: {', '.join(wrong[:3])}. "
        "The line in desktop/build/build_runtime.py (RELOCATION) that renames it leans on "
        "zipimport._unmarshal_code and _imp._fix_co_filename, which this Python has "
        "changed or dropped: read zipimport.py of this version and update the line."
    ]


def describe(exc: BaseException | None) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:300]


# --- 4: the zipped packages ---------------------------------------------------


def to_be_zipped(runtime: Path) -> list[str]:
    sys.path.insert(0, str(BUILD))
    import site_zip

    return site_zip.plan(runtime / PACKAGES).modules


def zipped(runtime: Path) -> list[str]:
    sys.path.insert(0, str(BUILD))
    import site_zip

    return [unit.replace("/", ".") for unit in site_zip.packed(runtime)]


def zipped_imports(runtime: Path, record: Path | None) -> list[str]:
    packages = zipped(runtime)
    if not packages:
        print(f"  no {ARCHIVES}/: nothing zipped to check")
        return []
    now = import_packages(runtime, packages)
    if record is None or not record.is_file():
        print(
            f"  {len(now)} modules in {ARCHIVES}/, {sum(o != 'ok' for o in now.values())} do not "
            "import; NOT COMPARED with how they imported as files (no record: the zip step "
            "writes one when it starts from an unzipped bundle)"
        )
        return []
    before = json.loads(record.read_text(encoding="utf-8"))
    changed = sorted(
        name for name in before.keys() | now.keys() if before.get(name) != now.get(name)
    )
    print(
        f"  {len(now)} modules in {ARCHIVES}/, {len(changed)} import differently than as files"
    )
    return [
        *[
            f"{name} imported as a file with `{before.get(name, 'was not there')}` and from "
            f"its zip with `{now.get(name, 'is not there')}`"
            for name in changed
        ],
        *not_importing(packages, now),
    ]


def not_importing(packages: list[str], outcomes: dict[str, str]) -> list[str]:
    """A package that fails as it is imported has had none of its modules
    tried, as files or from the zip: the comparison says nothing about it."""
    failing = {
        name: outcomes[name] for name in packages if outcomes.get(name, "ok") != "ok"
    }
    return [
        *[
            f"the zipped package {name} does not import ({how}), so none of its modules "
            "was compared with how it imported as a file. See what it needs that a "
            "service has and this check does not; or, when it cannot import in the bundle, "
            "look at its modules by hand and put it on MAY_NOT_IMPORT in build/bundle_gate.py "
            "with the reason"
            for name, how in sorted(failing.items())
            if name not in MAY_NOT_IMPORT
        ],
        *[
            f"{name} imports now: take it off MAY_NOT_IMPORT in build/bundle_gate.py"
            for name in sorted(MAY_NOT_IMPORT)
            if outcomes.get(name) == "ok"
        ],
    ]


def import_packages(runtime: Path, packages: list[str]) -> dict[str, str]:
    """module -> "ok" or how it failed (`outcome`), for every module of
    `packages`. Each package in an interpreter of its own: importing
    everything a package has can end the process, and must not end this."""
    outcomes: dict[str, str] = {}
    workers = max(2, (os.cpu_count() or 2) - 1)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for package, found in zip(
            packages,
            pool.map(lambda p: _in_a_process(runtime, p), packages),
            strict=True,
        ):
            outcomes.update(found or {package: "the process importing it ended"})
    return outcomes


def _in_a_process(runtime: Path, package: str) -> dict[str, str] | None:
    command = [sys.executable, "-B", __file__, str(runtime), "--package", package]
    with tempfile.TemporaryDirectory(prefix="autogpt-gate-") as scratch:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=PACKAGE_SECONDS,
                cwd=scratch,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONWARNINGS": "ignore",
                },
            )
        except subprocess.TimeoutExpired:
            return None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


def import_package(package: str) -> dict[str, str]:
    """Import the package and every module under it, wherever the
    interpreter finds it (site/ before zipping, site-zip/ after)."""
    return {name: outcome(exc) for name, exc in import_all(package).items()}


def outcome(exc: BaseException | None) -> str:
    """How an import ended, in words that are the same from a file and from
    a zip: the exception's type, and for an import that failed the module
    that could not be imported. Not the message: it names paths, which
    differ. A module that failed for want of one dependency and now fails
    for want of another has changed."""
    if exc is None:
        return "ok"
    missing = exc.name if isinstance(exc, ImportError) else None
    return f"{type(exc).__name__}: {missing}" if missing else type(exc).__name__


def import_all(package: str) -> dict[str, BaseException | None]:
    """module -> what importing it raised, for `package` and every module
    under it. Not pkgutil.walk_packages: it imports each package itself, and
    one that ends the interpreter as it is imported would end the walk."""
    outcomes: dict[str, BaseException | None] = {}

    def attempt(name: str) -> object | None:
        try:
            module = importlib.import_module(name)
        except BaseException as exc:
            outcomes[name] = exc
            return None
        outcomes[name] = None
        return module

    def below(module: object | None, prefix: str) -> None:
        for found in pkgutil.iter_modules(
            getattr(module, "__path__", None) or [], prefix
        ):
            # A __main__ module is a program; importing it is running it.
            if found.name.endswith(".__main__"):
                continue
            imported = attempt(found.name)
            if found.ispkg:
                below(imported, f"{found.name}.")

    below(attempt(package), f"{package}.")
    return outcomes


if __name__ == "__main__":
    sys.exit(main())
