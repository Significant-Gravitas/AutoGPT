"""What is left out of the bundle, what is zipped, and the budgets it is held to.

Three kinds of test, as in test_bundle_integrity.py: the rules themselves
against small made-up trees; the lists the rules lean on against the
backend's own source and lock, each failing with what to re-read when
upstream moves; and one zip made and imported from for real.

The build's own gates (build/bundle_gate.py) and the smoke test are what
check an assembled bundle. Nothing here needs one.
"""

import ast
import importlib
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[2]
PLATFORM = DESKTOP.parent
BACKEND = PLATFORM / "backend" / "backend"
LOCK = PLATFORM / "backend" / "poetry.lock"
sys.path.insert(0, str(DESKTOP / "build"))
backend_tests = importlib.import_module("backend_tests")
build_runtime = importlib.import_module("build_runtime")
bundle_gate = importlib.import_module("bundle_gate")
lock_export = importlib.import_module("lock_export")
site_zip = importlib.import_module("site_zip")
slim = importlib.import_module("slim")


def write(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def names(paths, root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in paths)


# --- the backend's tests -------------------------------------------------------


def backend(tmp_path: Path, files: dict[str, str]) -> Path:
    return write(tmp_path / "backend", {"__init__.py": "", **files})


def test_a_test_is_left_out_and_a_module_named_like_one_but_used_is_kept(
    tmp_path: Path,
):
    package = backend(
        tmp_path,
        {
            "rest.py": "from backend.util.test import wait\nfrom . import helpers_test\n",
            "util/__init__.py": "",
            "util/test.py": "def wait(): pass\n",  # named like nothing pytest collects
            "util/cache_test.py": "import pytest\n",
            "helpers_test.py": "from backend.fixtures import conftest\n",
            "fixtures/__init__.py": "",
            "fixtures/conftest.py": "",  # kept: through a test module that is kept
            "test_rest.py": "import backend.rest\n",
            "blocks/__init__.py": "",
            "blocks/exa/__init__.py": "",
            "blocks/exa/_test.py": "",  # a block's helper, by its name alone
            "blocks/exa/search_test.py": "import pytest\n",
            "blocks/test/test_block.py": "",
        },
    )

    removed = backend_tests.prune(package)

    assert names(removed, package) == [
        "blocks/exa/search_test.py",
        "blocks/test/test_block.py",
        "test_rest.py",
        "util/cache_test.py",
    ]
    assert not (package / "blocks" / "test").exists(), "a directory left empty goes too"
    for kept in (
        "util/test.py",
        "helpers_test.py",
        "fixtures/conftest.py",
        "blocks/exa/_test.py",
    ):
        assert (package / kept).is_file(), kept


def test_a_test_that_another_module_names_in_a_string_is_kept(tmp_path: Path):
    """How a dynamic import would name it."""
    package = backend(
        tmp_path,
        {
            "loader.py": 'import importlib\nimportlib.import_module("backend.data_test")\n',
            "data_test.py": "",
        },
    )
    assert backend_tests.removable(package) == []


def test_bytecode_of_a_removed_test_goes_with_it(tmp_path: Path):
    package = backend(
        tmp_path, {"a_test.py": "", "__pycache__/a_test.cpython-313.pyc": "x"}
    )
    backend_tests.prune(package)
    assert not (package / "__pycache__").exists()


def test_a_test_file_that_defines_a_block_stops_the_build(tmp_path: Path):
    """The block loader skips only `test_*`: it would have registered it."""
    package = backend(
        tmp_path,
        {
            "blocks/__init__.py": "",
            "blocks/maths_test.py": "from backend.blocks._base import Block\nclass ProbeBlock(Block): pass\n",
        },
    )
    with pytest.raises(backend_tests.BlockInTestError, match="defines a block"):
        backend_tests.prune(package)
    assert (package / "blocks" / "maths_test.py").is_file()


def test_the_block_loader_still_imports_every_file_but_test_prefixed_ones():
    loader = (BACKEND / "blocks" / "__init__.py").read_text(encoding="utf-8")
    assert 'f.name.startswith("test_")' in loader and 'rglob("*.py")' in loader, (
        "backend/blocks/__init__.py no longer loads blocks by importing every file under "
        "blocks/ except test_*. desktop/build/backend_tests.py leaves `*_test.py` files there "
        "out of the bundle because that loader imported them at every start: read the new "
        "loader and see whether it still must, and whether a file left out is one it needs."
    )


def test_what_is_left_out_of_the_real_backend_is_tests_and_nothing_a_service_imports():
    removable = backend_tests.removable(BACKEND)
    kept_tests = {
        path.relative_to(BACKEND).as_posix()
        for path in BACKEND.rglob("*.py")
        if backend_tests.TEST_NAME.fullmatch(path.name) and path not in set(removable)
    }
    assert (
        len(removable) > 500
    ), "the backend's tests are no longer named *_test.py or test_*.py"
    assert all(backend_tests.TEST_NAME.fullmatch(path.name) for path in removable)
    # Named like tests, imported by modules that are not: these must stay.
    assert "api/features/admin/test_data_routes.py" in kept_tests
    for helper in ("blocks/exa/_test.py", "util/test.py"):
        assert (
            BACKEND / helper
        ).is_file() and BACKEND / helper not in removable, (
            f"backend/{helper} is gone or would be left out; it is imported at start"
        )


# --- what the lock lists and the bundle does not install ---------------------


def top_level_imports(package: Path, skip) -> dict[str, str]:
    """Every top-level name some module of the backend imports -> one module
    that does."""
    found: dict[str, str] = {}
    for path in package.rglob("*.py"):
        if skip(path):
            continue
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        ):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            for module in modules:
                found.setdefault(
                    module.split(".")[0], path.relative_to(package).as_posix()
                )
    return found


def test_no_backend_module_that_ships_imports_what_is_left_out():
    removed = set(backend_tests.removable(BACKEND))
    imported = top_level_imports(BACKEND, lambda path: path in removed)
    left_out = {name.replace("-", "_") for name in lock_export.excluded(LOCK)}
    used = {name: imported[name] for name in left_out & imported.keys()}
    assert not used, (
        f"the backend now imports {used}, which desktop/build/lock_export.py leaves out of the "
        "bundle with EXCLUDED_ROOTS. Take the root that brings it off that list."
    )


def test_poetry_and_flake8_are_left_out_and_what_the_backend_needs_is_not():
    left_out = set(lock_export.excluded(LOCK))
    assert {"poetry", "flake8", "virtualenv", "dulwich"} <= left_out
    kept = {line.split("==")[0] for line in lock_export.export(LOCK)}
    # aioclamd imports pkg_resources as it loads, and the backend imports it;
    # backend modules that are not tests import pytest.
    for needed in (
        "setuptools",
        "aioclamd",
        "pytest",
        "pytest-asyncio",
        "pytest-snapshot",
    ):
        assert needed in kept, f"{needed} is no longer installed in the bundle"
    assert not left_out & kept


def test_modules_outside_the_tests_still_import_pytest_which_is_why_it_ships():
    removed = set(backend_tests.removable(BACKEND))
    importers = top_level_imports(BACKEND, lambda path: path in removed)
    assert "pytest" in importers, (
        "no backend module that ships imports pytest any more. pytest, pytest-asyncio and "
        "pytest-snapshot can join EXCLUDED_ROOTS in desktop/build/lock_export.py (the desktop's "
        "own tests would then need pytest from elsewhere: README, development loop)."
    )


def test_seeded_pip_is_not_a_locked_dependency_or_a_runtime_import():
    assert not any(
        re.match(r"pip(?:\s|[<>=!~;\[]|$)", requirement, re.IGNORECASE)
        for requirement in lock_export.export(LOCK)
    )
    removed = set(backend_tests.removable(BACKEND))
    for package in (
        BACKEND,
        PLATFORM / "autogpt_libs" / "autogpt_libs",
        DESKTOP / "runtime" / "autogpt_desktop",
    ):
        assert "pip" not in top_level_imports(package, lambda path: path in removed)


def test_seeded_pip_and_its_metadata_go_without_other_packages(tmp_path: Path):
    site = write(
        tmp_path / "site",
        {
            "pip/__init__.py": "",
            "pip/_vendor/library.py": "",
            "pip/__pycache__/__init__.cpython-313.pyc": "",
            "pip-25.2.dist-info/METADATA": "Name: pip\n",
            "setuptools/__init__.py": "",
            "pkg_resources/__init__.py": "",
            "pip_tools/__init__.py": "",
            "pip_tools-1.0.dist-info/METADATA": "Name: pip-tools\n",
        },
    )
    unused = slim.unused_pip(site, ["setuptools==80.9.0", "pip-tools==1.0"])
    assert names(unused, site) == ["pip", "pip-25.2.dist-info"]
    assert slim.remove(unused) == 4
    assert slim.unused_pip(site, []) == []
    assert (site / "setuptools" / "__init__.py").is_file()
    assert (site / "pkg_resources" / "__init__.py").is_file()
    assert (site / "pip_tools" / "__init__.py").is_file()
    assert (site / "pip_tools-1.0.dist-info" / "METADATA").is_file()


@pytest.mark.parametrize(
    "requirement",
    ["pip==25.2", "Pip>=25", "pip ; python_version > '3'", "pip @ file:///wheel"],
)
def test_a_locked_pip_is_never_pruned(tmp_path: Path, requirement: str):
    site = write(tmp_path / "site", {"pip/__init__.py": ""})
    assert slim.unused_pip(site, [requirement]) == []


def test_seal_rejects_leftover_seeded_pip(tmp_path: Path):
    out = write(
        tmp_path / "runtime",
        {"site/pip/__init__.py": "", "site/pip-25.2.dist-info/METADATA": "Name: pip\n"},
    )
    assert names(build_runtime.never_shipped(out), out) == [
        "site/pip",
        "site/pip-25.2.dist-info",
    ]


@pytest.mark.parametrize("relocated", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_prune_handles_seeded_pip_before_and_after_relocation(
    tmp_path: Path, monkeypatch, relocated: bool, required: bool
):
    build = build_runtime.Build(tmp_path / "runtime", tmp_path / "cache")
    write(
        build.out,
        {
            "python/python.exe": "",
            "erlang/lib/stdlib-1/ebin/stdlib.app": "",
            "rabbitmq/plugins/rabbit-4.3.6/ebin/rabbit.app": (
                "{application, rabbit, [{applications, [rabbitmq_prelaunch]}]}."
            ),
            "rabbitmq/plugins/rabbitmq_prelaunch-4.3.6/ebin/rabbitmq_prelaunch.app": (
                "{application, rabbitmq_prelaunch, [{applications, []}]}."
            ),
            "rabbitmq/escript/rabbitmqctl": "",
        },
    )
    site = write(
        build.out / "site" if relocated else build.site_packages,
        {
            "pip/__init__.py": "",
            "pip-25.2.dist-info/METADATA": "Name: pip\n",
            "setuptools/__init__.py": "",
            "pkg_resources/__init__.py": "",
        },
    )
    requirements = ["setuptools==80.9.0", *(["pip==25.2"] if required else [])]
    monkeypatch.setattr(lock_export, "export", lambda _: requirements)

    build.step_prune()

    assert (site / "pip" / "__init__.py").is_file() == required
    assert (site / "pip-25.2.dist-info" / "METADATA").is_file() == required
    assert (site / "setuptools" / "__init__.py").is_file()
    assert (site / "pkg_resources" / "__init__.py").is_file()
    assert build_runtime.never_shipped(build.out) == []


def test_a_root_that_is_no_longer_a_dependency_is_said_so(tmp_path: Path):
    project = {
        "tool": {
            "poetry": {
                "dependencies": {"python": "^3.13", "fastapi": "*", "flake8": "*"}
            }
        }
    }
    with pytest.raises(RuntimeError, match="poetry is no longer a main dependency"):
        lock_export.roots(project)


def test_what_only_an_excluded_root_needs_goes_and_what_is_shared_stays():
    lock = {
        "package": [
            {"name": "poetry", "dependencies": {"cleo": "*", "Requests": "*"}},
            {"name": "cleo", "dependencies": {"crashtest": "*"}},
            {"name": "crashtest"},
            {"name": "requests", "dependencies": {"urllib3": "*"}},
            {"name": "urllib3"},
            {
                "name": "fastapi",
                "dependencies": {"requests": {"version": "*", "optional": True}},
            },
        ]
    }
    assert lock_export.reachable(lock, {"fastapi"}) == {
        "fastapi",
        "requests",
        "urllib3",
    }


def test_a_package_the_lock_lists_twice_depends_on_what_either_entry_does():
    """The lock has an entry for each Python version or platform a package
    differs on (numpy has two). With the last one standing for both, what
    only the first needs was left out and uninstalled."""
    lock = {
        "package": [
            {"name": "fastapi", "dependencies": {"numpy": "*"}},
            {
                "name": "numpy",
                "version": "1.0",
                "dependencies": {"only-the-first": "*"},
            },
            {
                "name": "numpy",
                "version": "2.0",
                "dependencies": {"only-the-second": "*"},
            },
            {"name": "numpy", "version": "3.0"},
            {"name": "only-the-first"},
            {"name": "only-the-second"},
        ]
    }
    assert lock_export.reachable(lock, {"fastapi"}) == {
        "fastapi",
        "numpy",
        "only-the-first",
        "only-the-second",
    }


def test_setuptools_is_a_main_dependency_because_aioclamd_imports_pkg_resources():
    scanner = (BACKEND / "util" / "virus_scanner.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*import aioclamd$", scanner, re.M), (
        "backend/util/virus_scanner.py no longer imports aioclamd at module level. If nothing "
        "else imports pkg_resources, setuptools could be left out of the bundle "
        "(desktop/build/lock_export.py) and out of ON_DISK in desktop/build/site_zip.py."
    )


# --- third-party test directories ---------------------------------------------


def test_test_directories_go_unless_code_outside_them_imports_them(tmp_path: Path):
    site = write(
        tmp_path / "site",
        {
            "pandas/__init__.py": "",
            "pandas/tests/__init__.py": "",
            "pandas/tests/test_frame.py": "import pandas.tests.helpers\n",
            "numpy/__init__.py": "from numpy.testing import assert_equal\n",
            "numpy/testing/__init__.py": "",  # not a test directory by name
            "numpy/tests/test_core.py": "",
            "jsonschema/__init__.py": "",
            "jsonschema/benchmarks/issue.py": "from jsonschema.tests._suite import Suite\n",
            "jsonschema/tests/_suite.py": "",
            "relative/__init__.py": "from .test import helper\n",
            "relative/test/__init__.py": "",
            "other/__init__.py": "from . import tests\n",
            "other/tests/__init__.py": "",
            "plain/__init__.py": "import plain.test as t, os\n",
            "plain/test/__init__.py": "",
        },
    )

    unused = slim.unused_test_directories(site)

    assert names(unused, site) == ["numpy/tests", "pandas/tests"]


def test_build_only_files_are_stubs_sources_and_link_libraries(tmp_path: Path):
    site = write(
        tmp_path / "site",
        {
            "pkg/__init__.py": "",
            "pkg/__init__.pyi": "",
            "pkg/py.typed": "",
            "pkg/_speedups.c": "",
            "pkg/_speedups.pyx": "",
            "pkg/include/pkg.h": "",
            "pkg/libs/pkg.lib": "",
            "pkg/data.json": "{}",
            "pkg/_speedups.pyd": "",
            "pkg/cacert.pem": "",
        },
    )
    assert names(slim.build_only_files(site), site) == [
        "pkg/__init__.pyi",
        "pkg/_speedups.c",
        "pkg/_speedups.pyx",
        "pkg/include/pkg.h",
        "pkg/libs/pkg.lib",
        "pkg/py.typed",
    ]


# --- Google API descriptions ---------------------------------------------------

# What the backend builds clients for today. Not a list the build uses: the
# build reads the backend. It is here so that a call written in a way the
# reader does not see shows up as a missing API, not as a block that fails.
GOOGLE_APIS = {
    ("calendar", "v3"),
    ("docs", "v1"),
    ("drive", "v3"),
    ("gmail", "v1"),
    ("sheets", "v4"),
}


def test_every_google_api_the_backend_builds_a_client_for_is_found():
    found = slim.google_apis([BACKEND])
    assert found >= GOOGLE_APIS, (
        f"desktop/build/slim.py no longer finds the backend's calls for {sorted(GOOGLE_APIS - found)}: "
        "their descriptions would be deleted from the bundle and the blocks would fail"
    )
    # Every call there is, by a cruder reading of the same files.
    crude = set()
    for path in BACKEND.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "googleapiclient" in text:
            crude |= set(
                re.findall(r"""\bbuild\(\s*["'](\w+)["']\s*,\s*["'](\w+)["']""", text)
            )
    assert found == crude, (
        f"the backend builds Google API clients that desktop/build/slim.py does not see "
        f"({sorted(crude ^ found)}): read how they are called and teach google_apis"
    )


def test_a_client_built_for_an_api_that_is_not_written_out_stops_the_build(
    tmp_path: Path,
):
    source = write(
        tmp_path / "backend",
        {
            "blocks/google.py": (
                "from googleapiclient.discovery import build\n"
                "def service(name):\n    return build(name, 'v1')\n"
            )
        },
    )
    with pytest.raises(slim.SlimError, match="not written out"):
        slim.google_apis([source])


def test_the_ways_a_client_is_built_are_all_read(tmp_path: Path):
    source = write(
        tmp_path / "backend",
        {
            "a.py": "from googleapiclient.discovery import build\nbuild('gmail', 'v1', credentials=None)\n",
            "b.py": "from googleapiclient.discovery import build as make\nmake(serviceName='drive', version='v3')\n",
            "c.py": "import googleapiclient.discovery\ngoogleapiclient.discovery.build('docs', 'v1')\n",
            "d.py": "from googleapiclient import discovery\ndiscovery.build('sheets', 'v4')\n",
            "e.py": "from other import build\n# googleapiclient is only mentioned here\nbuild(1, 2)\n",
        },
    )
    assert slim.google_apis([source]) == {
        ("gmail", "v1"),
        ("drive", "v3"),
        ("docs", "v1"),
        ("sheets", "v4"),
    }


def test_only_the_named_descriptions_are_kept_and_a_missing_one_stops_the_build(
    tmp_path: Path,
):
    site = write(
        tmp_path / "site",
        {
            "googleapiclient/discovery_cache/documents/gmail.v1.json": "{}",
            "googleapiclient/discovery_cache/documents/youtube.v3.json": "{}",
            "googleapiclient/sample_tools.py": "from googleapiclient import discovery\ndiscovery.build(name, version)\n",
        },
    )
    source = write(
        tmp_path / "backend",
        {
            "gmail.py": "from googleapiclient.discovery import build\nbuild('gmail', 'v1')\n"
        },
    )

    unused = slim.unused_discovery_documents(site, [source, site])
    assert names(unused, site) == [
        "googleapiclient/discovery_cache/documents/youtube.v3.json"
    ]

    write(
        source,
        {
            "drive.py": "from googleapiclient.discovery import build\nbuild('drive', 'v3')\n"
        },
    )
    with pytest.raises(slim.SlimError, match="drive.v3.json"):
        slim.unused_discovery_documents(site, [source])


def test_the_client_still_reads_an_api_description_from_that_folder():
    """google-api-python-client is not installed where the unit tests run;
    the lock says which version the bundle gets."""
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    locked = {package["name"] for package in lock["package"]}
    assert "google-api-python-client" in locked, (
        "the backend no longer depends on google-api-python-client: take the discovery "
        "documents out of desktop/build/slim.py and build_runtime.step_prune"
    )
    assert (
        str(slim.DISCOVERY_DOCUMENTS.as_posix())
        == "googleapiclient/discovery_cache/documents"
    )


# --- RabbitMQ's plugins --------------------------------------------------------


def app(name: str, applications: str, included: str = "") -> str:
    extra = f"{{included_applications, [{included}]}},\n" if included else ""
    return f"{{application, '{name}', [\n{{vsn, \"1\"}},\n{{applications, [{applications}]}},\n{extra}{{env, []}}]}}.\n"


def rabbitmq_tree(tmp_path: Path) -> Path:
    return write(
        tmp_path / "rabbitmq",
        {
            "plugins/README.txt": "",
            "plugins/rabbit-4.1.8/ebin/rabbit.app": app(
                "rabbit", "kernel,stdlib,rabbit_common,ra"
            ),
            "plugins/rabbit_common-4.1.8/ebin/rabbit_common.app": app(
                "rabbit_common", "kernel, 'thoas'"
            ),
            "plugins/thoas-1.2.1/ebin/thoas.app": app("thoas", "kernel"),
            "plugins/ra-2.16.13/ebin/ra.app": app("ra", "kernel", included="aten"),
            "plugins/aten-0.6.0/ebin/aten.app": app("aten", "kernel"),
            "plugins/rabbitmq_prelaunch-4.1.8/ebin/rabbitmq_prelaunch.app": app(
                "rabbitmq_prelaunch", "kernel"
            ),
            "plugins/rabbitmq_mqtt-4.1.8/ebin/rabbitmq_mqtt.app": app(
                "rabbitmq_mqtt", "rabbit,ranch"
            ),
            "plugins/ranch-2.2.0/ebin/ranch.app": app("ranch", "kernel"),
            "escript/rabbitmqctl": "",
            "escript/rabbitmq-plugins": "",
            "escript/rabbitmq-diagnostics": "",
        },
    )


def test_plugins_the_server_does_not_start_go_and_what_it_depends_on_stays(
    tmp_path: Path,
):
    rabbitmq = rabbitmq_tree(tmp_path)
    unused = slim.unused_rabbitmq_plugins(rabbitmq)
    assert names(unused, rabbitmq) == [
        "plugins/rabbitmq_mqtt-4.1.8",
        "plugins/ranch-2.2.0",
    ]


def test_a_rabbitmq_that_is_laid_out_differently_stops_the_build(tmp_path: Path):
    rabbitmq = rabbitmq_tree(tmp_path)
    (rabbitmq / "plugins" / "rabbit-4.1.8").rename(
        rabbitmq / "plugins" / "rabbitmq_server-4.1.8"
    )
    with pytest.raises(slim.SlimError, match="no rabbit under plugins"):
        slim.unused_rabbitmq_plugins(rabbitmq)


def test_only_the_tool_the_runtime_runs_is_kept(tmp_path: Path):
    rabbitmq = rabbitmq_tree(tmp_path)
    assert names(slim.unused_rabbitmq_tools(rabbitmq), rabbitmq) == [
        "escript/rabbitmq-diagnostics",
        "escript/rabbitmq-plugins",
    ]
    runtime = (DESKTOP / "runtime" / "autogpt_desktop" / "rabbitmq.py").read_text(
        encoding="utf-8"
    )
    tools = set(re.findall(r'_script\(bundle, "([\w-]+)"\)', runtime))
    assert tools == {"rabbitmq-server", slim.RABBITMQ_CLI_KEPT}, (
        f"the runtime now runs {sorted(tools)}; desktop/build/slim.py keeps only "
        f"escript/{slim.RABBITMQ_CLI_KEPT} of RabbitMQ's command-line tools"
    )


def test_what_goes_from_the_interpreter_and_what_the_build_still_needs_of_it(
    tmp_path: Path,
):
    python = write(
        tmp_path / "python",
        {
            "python.exe": "",
            "Lib/os.py": "",
            "Lib/tkinter/__init__.py": "",
            "Lib/idlelib/idle.py": "",
            "Lib/ensurepip/__init__.py": "",
            "Lib/json/__init__.py": "",
            "DLLs/_tkinter.pyd": "",
            "DLLs/tcl86t.dll": "",
            "DLLs/_ssl.pyd": "",
            "tcl/tcl8.6/init.tcl": "",
            "libs/python313.lib": "",
            "Scripts/poetry.exe": "",
            "Scripts/prisma-client-py.exe": "",
            "Scripts/prisma.exe": "",
        },
    )
    extras = names(slim.interpreter_extras(python), python)
    assert extras == [
        "DLLs/_tkinter.pyd",
        "DLLs/tcl86t.dll",
        "Lib/ensurepip",
        "Lib/idlelib",
        "Lib/tkinter",
        "Scripts/poetry.exe",
        "libs",
        "tcl",
    ]


def test_a_posix_interpreter_keeps_its_own_programs(tmp_path: Path):
    python = write(
        tmp_path / "python",
        {
            "bin/python3": "",
            "bin/python3.13": "",
            "bin/pip3": "",
            "bin/prisma-client-py": "",
            "lib/python3.13/os.py": "",
            "lib/python3.13/tkinter/__init__.py": "",
            "lib/python3.13/lib-dynload/_tkinter.cpython-313-darwin.so": "",
            "lib/python3.13/lib-dynload/_ssl.cpython-313-darwin.so": "",
            "lib/libpython3.13.dylib": "",
            "lib/tcl8.6/init.tcl": "",
            "lib/libtk8.6.dylib": "",
            "share/man/man1/python3.1": "",
        },
    )
    assert names(slim.interpreter_extras(python), python) == [
        "bin/pip3",
        "lib/libtk8.6.dylib",
        "lib/python3.13/lib-dynload/_tkinter.cpython-313-darwin.so",
        "lib/python3.13/tkinter",
        "lib/tcl8.6",
        "share",
    ]


# --- what is zipped ------------------------------------------------------------


def site_tree(tmp_path: Path) -> Path:
    return write(
        tmp_path / "site",
        {
            "pure/__init__.py": "VALUE = 1\n",
            "pure/sub/mod.py": "def where():\n    raise RuntimeError('from the zip')\n",
            "native/__init__.py": "",
            "native/_speedups.pyd": "",
            "certifi/__init__.py": "",
            "certifi/cacert.pem": "",
            "tzdata/__init__.py": "",
            "tzdata/zoneinfo/Europe/Paris": "TZif",
            "setuptools/__init__.py": "",
            "pkg_resources/__init__.py": "",
            "_distutils_hack/__init__.py": "",
            "distutils-precedence.pth": "import os; enabled = True; __import__('_distutils_hack')\n",
            "pywin32.pth": "win32\nwin32\\lib\nimport pywin32_bootstrap\n",
            "win32/lib/win32con.py": "",
            "win32comext/axdebug/__init__.py": "",
            "pywin32_bootstrap.py": "",
            "pywin32-311.dist-info/RECORD": "win32comext/axdebug/__init__.py,sha256=x,1\npythoncom.py,,\n",
            "single.py": "",
            "pure-1.0.dist-info/METADATA": "Name: pure\n",
            "google/auth/__init__.py": "",
            "google/auth/transport.py": "",
            "google/protobuf/__init__.py": "",
            "google/_upb/_message.pyd": "",
            "google/api/annotations_pb2.py": "",
            "google/api/annotations.proto": "",
            "google/cloud/storage/__init__.py": "",
            "google/cloud/logging/__init__.py": "",
            "google/cloud/logging/gapic_metadata.json": "{}",
            "opentelemetry/trace/__init__.py": "",
        },
    )


def test_a_package_is_zipped_whole_or_not_at_all(tmp_path: Path):
    chosen = site_zip.plan(site_tree(tmp_path))

    assert chosen.zipped == (
        "google/auth",
        "google/cloud/storage",
        "google/protobuf",
        "opentelemetry/trace",
        "pure",
        "tzdata",
    )
    assert chosen.modules[:2] == ["google.auth", "google.cloud.storage"]
    assert set(chosen.kept) == {
        "_distutils_hack",
        "certifi",
        "google/_upb",
        "google/api",
        "google/cloud/logging",
        "native",
        "pkg_resources",
        "setuptools",
        "win32",
        "win32comext",
    }
    assert "cacert.pem" in chosen.kept["certifi"]
    assert (
        "pywin32" not in chosen.kept["win32comext"]
        and "by path" in chosen.kept["win32comext"]
    )


# --- a package that names its own files ----------------------------------------


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("import os\nHERE = os.path.dirname(__file__)\n", True),
        ("import other\nprint(other.__file__)\n", True),
        ("import pkgutil\nfound = pkgutil.iter_modules(__path__)\n", True),
        (
            "def version():\n    from pathlib import Path\n    return Path(__file__).parent\n",
            True,
        ),
        ("this is not Python 3 <<<\n", True),
        ("# __file__ in a comment\nNOTE = 'and __file__ in a string'\n", False),
        ("import types\nmodule = types.ModuleType('m')\nmodule.__path__ = []\n", False),
        ("if __name__ == '__main__':\n    print(open(__file__).read())\n", False),
        ("if '__main__' == __name__:\n    print(__file__)\nelse:\n    X = 1\n", False),
        ("if __name__ == '__main__':\n    pass\nelse:\n    HERE = __file__\n", True),
    ],
)
def test_what_counts_as_a_module_naming_its_own_file(source: str, names: bool):
    assert site_zip.names_own_file(source) is names


def test_a_package_that_names_its_own_files_stays_on_disk(tmp_path: Path):
    """firecrawl read its version out of `<package>/__init__.py` by path.
    From a zip that failed, was swallowed, and every request went out as
    `python-sdk@3.x.x`; every module still imported, so no gate saw it."""
    site = write(
        tmp_path / "site",
        {
            "reads/__init__.py": "",
            "reads/version.py": "from pathlib import Path\nTEXT = (Path(__file__).parent / '__init__.py').read_text()\n",
            "program/__init__.py": "",
            "program/__main__.py": "print(__file__)\n",
            "plain/__init__.py": "",
            "google/reads/__init__.py": "HERE = __file__\n",
            "google/plain/__init__.py": "",
        },
    )

    chosen = site_zip.plan(site)

    assert chosen.zipped == ("google/plain", "plain", "program")
    assert chosen.kept == {
        "google/reads": "names its own files (google/reads/__init__.py)",
        "reads": "names its own files (reads/version.py)",
    }


def test_a_package_whose_uses_were_read_is_zipped_and_a_new_use_stops_the_build(
    tmp_path: Path, monkeypatch
):
    site = write(
        tmp_path / "site",
        {
            "sdk/__init__.py": "",
            "sdk/frames.py": "HERE = __file__\n",
            "sdk/client.py": "",
        },
    )
    monkeypatch.setitem(
        site_zip.NAMES_ITS_FILES,
        "sdk",
        {"sdk/frames.py": "tells its frames by directory"},
    )
    assert site_zip.plan(site).zipped == ("sdk",)

    write(
        site,
        {
            "sdk/client.py": "import os\nVERSION = open(os.path.join(os.path.dirname(__file__), 'v')).read()\n"
        },
    )
    with pytest.raises(site_zip.ZipError) as stopped:
        site_zip.plan(site)
    assert "sdk/client.py of sdk names its own file" in str(stopped.value)
    assert "NAMES_ITS_FILES" in str(stopped.value) and "stays on disk" in str(
        stopped.value
    )


# The packages whose every use of `__file__` was read before they were zipped,
# and the modules that were read. Adding to either is a claim that someone
# read the new use and that it opens nothing beside the module by path.
READ_BEFORE_ZIPPING = {
    "anyio": {"anyio/_lazyimport.py"},
    "e2b": {"e2b/template/utils.py"},
    "imageio": {
        "imageio/core/findlib.py",
        "imageio/core/util.py",
        "imageio/plugins/pillow_info.py",
        "imageio/testing.py",
    },
    "neo4j": {"neo4j/_async/work/result.py", "neo4j/_sync/work/result.py"},
    "posthog": {"posthog/exception_utils.py"},
    "pydantic": {"pydantic/version.py", "pydantic/v1/version.py"},
    "pygments": {"pygments/sphinxext.py"},
    "rich": {"rich/pretty.py", "rich/traceback.py"},
    "sentry_sdk": {"sentry_sdk/utils.py"},
    "yt_dlp": {
        "yt_dlp/__pyinstaller/__init__.py",
        "yt_dlp/plugins.py",
        "yt_dlp/update.py",
    },
}


def test_the_packages_zipped_though_they_name_their_files_are_the_ones_that_were_read():
    listed = {
        package: set(modules) for package, modules in site_zip.NAMES_ITS_FILES.items()
    }
    assert listed == READ_BEFORE_ZIPPING
    assert all(
        why for modules in site_zip.NAMES_ITS_FILES.values() for why in modules.values()
    )
    assert (
        "firecrawl" in site_zip.ON_DISK and "firecrawl" not in site_zip.NAMES_ITS_FILES
    )


# --- a package zipped with its data --------------------------------------------


def test_a_listed_package_with_a_file_nobody_looked_at_stops_the_build(tmp_path: Path):
    """anthropic once shipped a tokenizer.json and opened it by path. Added
    again, it would be zipped without a word and fail on the user's machine:
    every module still imports."""
    site = write(
        tmp_path / "site",
        {
            "anthropic/__init__.py": "",
            "anthropic/lib/.keep": "",
            "anthropic/lib/foundry.md": "",
        },
    )
    assert site_zip.plan(site).zipped == ("anthropic",)

    write(site, {"anthropic/tokenizer.json": "{}"})
    with pytest.raises(site_zip.ZipError) as stopped:
        site_zip.plan(site)
    assert "anthropic has files" in str(stopped.value) and "tokenizer.json" in str(
        stopped.value
    )
    assert "DATA_READ_THROUGH_THE_LOADER" in str(stopped.value)


def test_a_listed_package_that_gains_a_compiled_module_stops_the_build(tmp_path: Path):
    """Whatever its entry's patterns match: a library is loaded by path."""
    site = write(
        tmp_path / "site",
        {
            "tzdata/__init__.py": "",
            "tzdata/zoneinfo/UTC": "TZif",
            "tzdata/zoneinfo/_speedups.pyd": "",
        },
    )
    with pytest.raises(site_zip.ZipError, match="zoneinfo/_speedups.pyd"):
        site_zip.plan(site)


def test_the_data_of_each_listed_package_is_named():
    for name, (why, patterns) in site_zip.DATA_READ_THROUGH_THE_LOADER.items():
        assert (
            why and patterns
        ), f"{name}: say what its other files are and how they are read"
        assert (
            "*" not in patterns
        ), f"{name}: a pattern that matches everything pins nothing"


# --- bytecode in the archives --------------------------------------------------


def test_an_archive_whose_modules_have_no_bytecode_stops_the_build(
    tmp_path: Path, monkeypatch
):
    """zipimport writes none: a module stored without it is compiled by every
    process of every start, and it imports, so nothing else would tell."""
    write(
        tmp_path / "site", {"pure/__init__.py": "VALUE = 1\n", "pure/mod.py": "X = 2\n"}
    )
    monkeypatch.setattr(
        site_zip, "_compile", lambda *args: None
    )  # the compile that did not run

    with pytest.raises(site_zip.ZipError) as stopped:
        site_zip.pack(tmp_path, Path(sys.executable))

    assert "2 modules are zipped without bytecode" in str(stopped.value)
    assert "pure/__init__.py" in str(stopped.value)


def test_a_file_that_is_not_python_3_may_be_without_bytecode(tmp_path: Path):
    write(
        tmp_path / "site",
        {
            "pure/__init__.py": "",
            "pure/py2.py": "print 'hello'\n",
            "pure/template.py": "{{ name }} = 1\n",
        },
    )

    site_zip.pack(tmp_path, Path(sys.executable))

    (archive,) = archives_of(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        assert (
            "pure/__init__.pyc" in bundle.namelist()
            and "pure/py2.pyc" not in bundle.namelist()
        )
    assert site_zip.without_bytecode(archive) == []


def test_bytecode_that_would_be_checked_against_its_source_is_not_accepted(
    tmp_path: Path,
):
    """Timestamp bytecode, as a plain compileall writes: zipimport compares
    it with the entry's time, and every archive entry has the same one."""
    archive = tmp_path / "00.zip"
    source = tmp_path / "mod.py"
    source.write_text("X = 1\n", encoding="utf-8")
    import py_compile

    compiled = py_compile.compile(
        str(source),
        cfile=str(tmp_path / "mod.pyc"),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source, "pure/mod.py")
        bundle.write(compiled, "pure/mod.pyc")

    assert site_zip.without_bytecode(archive) == [
        "pure/mod.pyc (its flags are 0, not unchecked-hash)"
    ]


# --- code from a zip names its file ---------------------------------------------


# A Python whose `_imp` has lost `_fix_co_filename`, as the line sees it: the
# import system keeps the module it started with.
WITHOUT_THE_NAME = "import types\nreal = sys.modules['_imp']\nsys.modules['_imp'] = types.ModuleType('_imp')\n"
WITH_IT_AGAIN = "sys.modules['_imp'] = real\n"


def run_with_the_relocation(
    tmp_path: Path, renaming: bool = True, then: str = ""
) -> subprocess.CompletedProcess:
    """An interpreter that reads RELOCATION as the bundle's does, over the
    made-up packages zipped under tmp_path. Started without `site` (-S): the
    interpreter these tests run on may be the bundle's, which has read the
    line already."""
    site = site_tree(tmp_path)
    site_zip.pack(site.parent, Path(sys.executable))
    packages = tmp_path / "python" / "Lib" / "site-packages"
    packages.mkdir(parents=True)
    (packages / build_runtime.RELOCATION_FILE).write_text(
        build_runtime.RELOCATION, encoding="utf-8"
    )
    code = (
        "import site, sys, traceback\n"
        f"{'' if renaming else WITHOUT_THE_NAME}"
        f"sys.prefix = {str(tmp_path / 'python')!r}\n"
        f"site.addpackage({str(packages)!r}, {build_runtime.RELOCATION_FILE!r}, set())\n"
        f"{'' if renaming else WITH_IT_AGAIN}"
        "import pure.sub.mod as module\n"
        "print(module.__file__)\n"
        "print(module.where.__code__.co_filename)\n"
        "try:\n    module.where()\nexcept RuntimeError:\n    print(traceback.format_exc())\n"
        f"{then}"
    )
    return subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env=_clean_environment(),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )


def test_code_loaded_from_a_zip_names_the_file_it_came_from(tmp_path: Path):
    """As code loaded from a .pyc file does. neo4j and e2b tell their own
    frames from a caller's by comparing file names with their directory."""
    result = run_with_the_relocation(tmp_path)

    assert result.returncode == 0 and result.stderr == "", result.stderr
    module_file, code_file = result.stdout.splitlines()[:2]
    archive = tmp_path / site_zip.ARCHIVES / site_zip.shard("pure")
    assert module_file == str(archive / "pure" / "sub" / "mod.pyc")
    assert code_file == str(archive / "pure" / "sub" / "mod.py"), (
        "build_runtime.RELOCATION no longer renames code loaded from an archive. It replaces "
        "zipimport._unmarshal_code(importer, path, ...) and calls _imp._fix_co_filename(code, "
        "path): read zipimport.py of this Python and update the line."
    )
    assert f'File "{code_file}", line 2, in where' in result.stdout
    assert "raise RuntimeError('from the zip')" in result.stdout


def test_a_python_without_the_names_that_line_uses_still_starts(tmp_path: Path):
    """Private names of CPython. Without them the code keeps the name it was
    compiled under, and the build's gate says so; nothing fails to import."""
    result = run_with_the_relocation(tmp_path, renaming=False)

    assert result.returncode == 0 and result.stderr == "", result.stderr
    assert result.stdout.splitlines()[1] == str(
        Path(site_zip.ARCHIVES) / "pure" / "sub" / "mod.py"
    )


ASK_THE_GATE = (
    f"sys.path.insert(0, {str(DESKTOP / 'build')!r})\n"
    "import bundle_gate\n"
    "print(bundle_gate.code_named_for_another_file(['pure', 'google.auth', 'opentelemetry']))\n"
)


def test_the_gate_says_when_zipped_code_is_not_renamed(tmp_path: Path):
    renamed = run_with_the_relocation(tmp_path / "renamed", then=ASK_THE_GATE)
    assert renamed.returncode == 0, renamed.stderr
    assert "3 of 3 zipped packages' code names its file" in renamed.stdout
    assert renamed.stdout.splitlines()[-1] == "[]"

    left = run_with_the_relocation(tmp_path / "left", renaming=False, then=ASK_THE_GATE)
    assert left.returncode == 0, left.stderr
    assert "1 of 3 zipped packages' code names its file" in left.stdout
    problem = left.stdout.splitlines()[-1]
    assert (
        "RELOCATION" in problem and "pure (" in problem and "google.auth (" in problem
    )


def archives_of(root: Path) -> list[Path]:
    return sorted((root / site_zip.ARCHIVES).glob("*.zip"))


def members_of(root: Path) -> dict[str, str]:
    """Every entry of every archive -> the archive it is in."""
    found: dict[str, str] = {}
    for archive in archives_of(root):
        with zipfile.ZipFile(archive) as bundle:
            found.update(dict.fromkeys(bundle.namelist(), archive.name))
    return found


def test_zipped_packages_import_from_the_zips_with_their_bytecode_and_data(
    tmp_path: Path,
):
    site = site_tree(tmp_path)
    root = site.parent

    chosen = site_zip.pack(root, Path(sys.executable))

    members = members_of(root)
    assert {
        "pure/__init__.py",
        "pure/__init__.pyc",
        "pure/sub/mod.pyc",
        "google/",
        "google/auth/",
    } <= set(members)
    assert "tzdata/zoneinfo/Europe/Paris" in members
    for archive in archives_of(root):
        with zipfile.ZipFile(archive) as bundle:
            assert {info.compress_type for info in bundle.infolist()} == {
                zipfile.ZIP_STORED
            }
    for unit in chosen.zipped:
        assert not (site / unit).exists(), f"{unit} is zipped and on disk"
        assert members[f"{unit}/"] == site_zip.shard(unit)
    assert (site / "google" / "api" / "annotations_pb2.py").is_file()
    assert not (
        site / "opentelemetry"
    ).exists(), "a namespace that was zipped whole leaves no directory"
    assert (site / "pure-1.0.dist-info" / "METADATA").is_file()
    assert site_zip.packed(root) == sorted(chosen.zipped)

    code = (
        "import sys, importlib.resources, traceback\n"
        "import pure.sub.mod, google.auth.transport, google.cloud.storage, google.protobuf\n"
        "import google.api.annotations_pb2, opentelemetry.trace\n"
        "print(pure.sub.mod.__file__)\n"
        "print(importlib.resources.files('tzdata').joinpath('zoneinfo/Europe/Paris').read_bytes())\n"
        "try:\n    pure.sub.mod.where()\nexcept RuntimeError:\n    print(traceback.format_exc())\n"
    )
    path = _pathsep().join(str(entry) for entry in [site, *archives_of(root)])
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-c", code],
        env={**_clean_environment(), "PYTHONPATH": path},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    said = result.stdout
    assert (
        str(root / site_zip.ARCHIVES / site_zip.shard("pure")) in said.splitlines()[0]
    )
    assert "b'TZif'" in said
    # The file a traceback names is the one in the zip, not the build's.
    assert f'{site_zip.ARCHIVES}{_sep()}pure{_sep()}sub{_sep()}mod.py", line 2' in said
    assert str(tmp_path) not in said.split("Traceback")[1]
    assert (
        "raise RuntimeError('from the zip')" in said
    ), "the source line is read from the zip"


def _sep() -> str:
    import os

    return os.sep


def _pathsep() -> str:
    import os

    return os.pathsep


def _clean_environment() -> dict[str, str]:
    import os

    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTHON")
    }


def test_bytecode_in_the_zip_is_used_without_a_look_at_the_source(tmp_path: Path):
    site = write(tmp_path / "site", {"pure/__init__.py": "VALUE = 'compiled'\n"})
    site_zip.pack(tmp_path, Path(sys.executable))
    (archive,) = archives_of(tmp_path)
    # The same zip with the source changed and the bytecode left alone.
    with zipfile.ZipFile(archive) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    members["pure/__init__.py"] = b"VALUE = 'source'\n"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)

    result = subprocess.run(
        [sys.executable, "-B", "-c", "import pure; print(pure.VALUE)"],
        env={**_clean_environment(), "PYTHONPATH": str(archive)},
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.stdout.strip() == "compiled", (
        "zipimport no longer trusts unchecked-hash bytecode beside its source: every import "
        "from site-zip/ recompiles. Read zipimport._unmarshal_code and update "
        "desktop/build/site_zip.py (_compile)."
    )
    assert site.exists()


def sizes(root: Path) -> dict[str, int]:
    return {archive.name: archive.stat().st_size for archive in archives_of(root)}


def test_packing_twice_changes_nothing_and_unpacking_gives_the_files_back(
    tmp_path: Path,
):
    site = site_tree(tmp_path)
    root = site.parent
    first = site_zip.pack(root, Path(sys.executable))
    packed = sizes(root)

    again = site_zip.pack(root, Path(sys.executable))
    assert again.zipped == () and sizes(root) == packed

    site_zip.unpack(root)
    assert not (root / site_zip.ARCHIVES).exists()
    assert (site / "pure" / "sub" / "mod.py").is_file()
    assert not list(site.rglob("*.pyc")), "bytecode is the compile step's to make"
    assert site_zip.plan(site).zipped == first.zipped


def test_a_package_zipped_and_on_disk_stops_the_build(tmp_path: Path):
    """Python would import one and never see the other."""
    site = site_tree(tmp_path)
    site_zip.pack(site.parent, Path(sys.executable))
    write(
        site, {"pure/__init__.py": "", "pure/new.py": ""}
    )  # installed again, beside the zips

    with pytest.raises(site_zip.ZipError, match="pure"):
        site_zip.pack(site.parent, Path(sys.executable))


def contents(root: Path) -> dict[str, list]:
    found = {}
    for archive in archives_of(root):
        with zipfile.ZipFile(archive) as bundle:
            found[archive.name] = [
                (i.filename, i.date_time, i.CRC) for i in bundle.infolist()
            ]
    return found


def test_an_archive_is_the_same_bytes_whenever_it_is_built(tmp_path: Path):
    """What a differential update compares."""
    built = []
    for name in ("one", "two"):
        site = site_tree(tmp_path / name)
        site_zip.pack(site.parent, Path(sys.executable))
        built.append(
            {archive.name: archive.read_bytes() for archive in archives_of(site.parent)}
        )
    assert built[0] == built[1]


def test_a_package_that_changes_takes_its_own_archive_with_it_and_no_other(
    tmp_path: Path,
):
    """An update downloads what differs in the installer, which compresses
    each file of the bundle by itself: a package that grows moves everything
    behind it in its archive. In one archive of everything, that was half of
    all the zipped packages for one changed line."""
    before = site_tree(tmp_path / "before")
    after = site_tree(tmp_path / "after")
    write(after, {"google/auth/transport.py": "# one more line\n"})
    for site in (before, after):
        site_zip.pack(site.parent, Path(sys.executable))

    old, new = contents(before.parent), contents(after.parent)

    changed = {
        name for name in old.keys() | new.keys() if old.get(name) != new.get(name)
    }
    assert changed == {site_zip.shard("google/auth")}
    assert len(old) > 1, "the packages of the made-up tree all fell into one archive"


SHARD_OF = {
    "openai": "15.zip",
    "yt_dlp": "10.zip",
    "google/auth": "26.zip",
    "pure": "29.zip",
}


def test_which_archive_a_package_is_in_depends_on_its_name_alone():
    """Change how it is chosen and the first update after that downloads
    every archive again."""
    assert site_zip.SHARDS == 32
    assert [
        site_zip.shard(unit) for unit in ("openai", "yt_dlp", "google/auth", "pure")
    ] == [
        SHARD_OF["openai"],
        SHARD_OF["yt_dlp"],
        SHARD_OF["google/auth"],
        SHARD_OF["pure"],
    ]


def locked_top_levels() -> set[str]:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    return {package["name"].lower().replace("-", "_") for package in lock["package"]}


# Top-level names in site_zip's lists whose distribution is called otherwise.
DISTRIBUTION_OF = {
    "markdown_it": "markdown_it_py",
    "firecrawl": "firecrawl_py",
    "pkg_resources": "setuptools",
    "_distutils_hack": "setuptools",
    "_pytest": "pytest",
    "autogpt_libs": None,  # the platform's own, installed from its directory
}


@pytest.mark.parametrize(
    "listed",
    [
        "DATA_READ_THROUGH_THE_LOADER",
        "NAMES_ITS_FILES",
        "ON_DISK",
        "ON_DISK_DISTRIBUTIONS",
    ],
)
def test_every_package_named_for_the_zip_is_still_one_the_backend_installs(listed: str):
    locked = locked_top_levels()
    for name in getattr(site_zip, listed):
        distribution = DISTRIBUTION_OF.get(name, name)
        assert distribution is None or distribution in locked, (
            f"{name} is on {listed} in desktop/build/site_zip.py, and the backend's lock no "
            "longer installs it: take it off the list"
        )


def test_nothing_is_on_both_of_the_zips_lists():
    assert not set(site_zip.DATA_READ_THROUGH_THE_LOADER) & set(site_zip.ON_DISK)
    # What the adversarial review of the plan required to stay files.
    assert {"setuptools", "pkg_resources", "prisma"} <= set(site_zip.ON_DISK)
    assert "pywin32" in site_zip.ON_DISK_DISTRIBUTIONS
    for never in ("certifi", "pytz", "googleapiclient", "stripe", "backend"):
        assert never not in site_zip.DATA_READ_THROUGH_THE_LOADER


def test_the_interpreter_is_pointed_at_site_and_at_the_zips(
    tmp_path: Path, monkeypatch, capsys
):
    """One line, run by `site` as the interpreter starts: by the function
    that runs it there, which reports a line that fails and carries on."""
    import site

    line = build_runtime.RELOCATION
    assert line.startswith("import ") and line.count("\n") == 1 and line.endswith("\n")
    prefix = tmp_path / "python"
    packages = prefix / "Lib" / "site-packages"
    packages.mkdir(parents=True)
    (packages / build_runtime.RELOCATION_FILE).write_text(line, encoding="utf-8")
    (tmp_path / "site").mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "path", list(sys.path))

    site.addpackage(str(packages), build_runtime.RELOCATION_FILE, set())
    assert (
        capsys.readouterr().err == ""
    ), "the line failed; site.py printed why and went on"
    assert str(tmp_path / "site") in sys.path
    zips = tmp_path / "site-zip"
    assert not [
        entry for entry in sys.path if entry.startswith(str(zips))
    ]  # there is none

    zips.mkdir()
    for name in ("07.zip", "00.zip"):
        (zips / name).write_bytes(b"")
    site.addpackage(str(packages), build_runtime.RELOCATION_FILE, set())
    assert capsys.readouterr().err == ""
    assert sys.path[-2:] == [str(zips / "00.zip"), str(zips / "07.zip")]
    assert sys.path.index(str(tmp_path / "site")) < sys.path.index(str(zips / "00.zip"))


def test_a_zipped_standard_library_is_put_back_as_files(tmp_path: Path):
    """A bundle built while the Windows standard library was zipped. With
    the archive beside the interpreter, an installed Python's search path is
    read from the registry (site_zip.py); the build undoes it."""
    home = write(tmp_path / "python", {"Lib/site-packages/autogpt-desktop.pth": ""})
    with zipfile.ZipFile(home / "python313.zip", "w") as bundle:
        bundle.writestr("os.py", "x = 1\n")
        bundle.writestr("os.pyc", b"bytecode")
        bundle.writestr("json/", b"")
        bundle.writestr("json/__init__.py", "")
        bundle.writestr("json/__init__.pyc", b"bytecode")

    assert site_zip.unpack_stdlib(home, (3, 13)) is True

    assert not (home / "python313.zip").exists()
    assert site_zip.zipped_stdlib(home) == []
    assert (home / "Lib" / "os.py").read_text() == "x = 1\n"
    assert (home / "Lib" / "json" / "__init__.py").is_file()
    assert not list((home / "Lib").rglob("*.pyc")), "the caller compiles it again"
    assert (home / "Lib" / "site-packages" / "autogpt-desktop.pth").is_file()
    # Again: nothing to put back.
    assert site_zip.unpack_stdlib(home, (3, 13)) is False


def test_nothing_zips_the_standard_library():
    assert not hasattr(site_zip, "pack_stdlib"), (
        "the standard library must stay as files: zipped beside the interpreter it makes "
        "Windows add an installed Python's paths from the registry (build/site_zip.py)"
    )
    build = (DESKTOP / "build" / "build_runtime.py").read_text("utf-8")
    assert "unpack_stdlib" in build and "zipped_stdlib" in build


def test_a_search_path_entry_outside_the_bundle_is_found(tmp_path: Path):
    runtime = tmp_path / "runtime"
    (runtime / "python" / "Lib").mkdir(parents=True)
    inside = [str(runtime / "python" / "Lib"), str(runtime / "site-zip" / "00.zip"), ""]
    assert bundle_gate.entries_outside(inside, runtime) == []
    system = str(tmp_path / "hostedtoolcache" / "Python" / "3.13.7" / "x64" / "Lib")
    assert bundle_gate.entries_outside([*inside, system], runtime) == [system]


# --- the budgets ----------------------------------------------------------------


def test_a_bundle_over_a_budget_is_refused_in_words(tmp_path: Path, monkeypatch):
    out = write(
        tmp_path / "runtime", {"a.txt": "1234567890", "deep/" + "x" * 40 + ".py": ""}
    )
    assert build_runtime.over_budget(out) == []

    monkeypatch.setattr(build_runtime, "MAX_FILES", 1)
    monkeypatch.setattr(build_runtime, "MAX_BYTES", 5)
    monkeypatch.setattr(build_runtime, "MAX_RELATIVE_PATH", 20)
    monkeypatch.setattr(build_runtime, "WINDOWS", True)
    over = build_runtime.over_budget(out)

    assert len(over) == 3
    assert "2 files, and the budget is 1" in over[0]
    assert "characters long, and the budget is 20" in over[1]
    assert "MB, and the budget is" in over[2]


def test_a_seal_says_where_the_files_and_the_bytes_are(tmp_path: Path, capsys):
    """Printed by every build: a budget that breaks is then explained by the
    same table in the log of the last build that passed."""
    out = write(
        tmp_path / "runtime",
        {
            "frontend/server.js": "x" * 2_000_000,
            "site/pandas/a.py": "",
            "site/pandas/b.py": "",
            "site/small/a.py": "",
            "site/single.py": "",
        },
    )

    assert build_runtime.over_budget(out) == []

    table = capsys.readouterr().out.splitlines()
    assert any(line.split() == ["frontend/", "1", "files", "2", "MB"] for line in table)
    assert any(line.split() == ["site/", "4", "files", "0", "MB"] for line in table)
    largest = [
        line.split()[0]
        for line in table
        if line.strip().startswith("site/") and line.count("/") == 2
    ]
    assert largest == ["site/pandas/", "site/small/"]


def test_the_budgets_are_the_ones_the_size_work_was_held_to():
    """Loosening one is a decision, made here."""
    assert build_runtime.MAX_RELATIVE_PATH == 130
    if sys.platform == "win32":
        assert build_runtime.MAX_FILES == 25_000
        assert build_runtime.MAX_BYTES == 1_600_000_000
    else:
        assert build_runtime.MAX_FILES == 30_000
        assert build_runtime.MAX_BYTES == 2_300_000_000


def test_current_source_count_fits_only_after_seeded_pip_is_removed(
    tmp_path: Path, monkeypatch
):
    """CI run 40352729690/job 121679690040, not a rebuilt bundle fixture."""
    monkeypatch.setattr(site_zip, "longest_path", lambda _: (130, "x" * 130))
    monkeypatch.setattr(build_runtime, "where_it_is", lambda _: [])
    monkeypatch.setattr(site_zip, "count", lambda _: (25_539 - 839, 1_547_000_000))
    assert build_runtime.over_budget(tmp_path) == []

    monkeypatch.setattr(site_zip, "count", lambda _: (25_539, 1_547_000_000))
    expected = ["25539 files, and the budget is 25000"] if build_runtime.WINDOWS else []
    assert build_runtime.over_budget(tmp_path) == expected


def test_file_budget_accepts_its_limit_and_rejects_one_more(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(site_zip, "longest_path", lambda _: (0, ""))
    monkeypatch.setattr(build_runtime, "where_it_is", lambda _: [])
    limit = build_runtime.MAX_FILES
    monkeypatch.setattr(site_zip, "count", lambda _: (limit, 0))
    assert build_runtime.over_budget(tmp_path) == []
    monkeypatch.setattr(site_zip, "count", lambda _: (limit + 1, 0))
    assert build_runtime.over_budget(tmp_path) == [
        f"{limit + 1} files, and the budget is {limit}"
    ]


def test_what_a_run_or_an_older_build_leaves_is_never_shipped(tmp_path: Path):
    out = write(
        tmp_path / "runtime",
        {
            "frontend/.next/cache/images/x.webp": "",
            "erlang/doc/index.html": "",
            "postgres/include/pg.h": "",
            "prisma/node_modules/@prisma/engines/node_modules/.cache/prisma/x.gz": "",
            "prisma/node_modules/@prisma/engines/query_engine-windows.dll.node": "",
            "prisma/query-engine.exe": "",
            "frontend/server.js": "",
        },
    )
    assert names(build_runtime.never_shipped(out), out) == [
        "erlang/doc",
        "frontend/.next/cache",
        "postgres/include",
        "prisma/node_modules/@prisma/engines/node_modules/.cache",
        "prisma/node_modules/@prisma/engines/query_engine-windows.dll.node",
    ]


def test_the_steps_run_in_an_order_that_zips_what_was_compiled_and_seals_last():
    steps = list(build_runtime.STEPS)
    for earlier, later in [
        ("deps", "backend"),
        ("backend", "prune"),
        ("prune", "relocate"),
        ("relocate", "compile"),
        ("compile", "zip"),
        ("zip", "verify"),
        ("verify", "seal"),
    ]:
        assert steps.index(earlier) < steps.index(
            later
        ), f"{earlier} must run before {later}"
    assert steps[-1] == "seal"


# --- the gate -------------------------------------------------------------------


def test_the_gate_imports_every_module_of_a_package_and_says_how_each_failed(
    tmp_path: Path, monkeypatch
):
    write(
        tmp_path,
        {
            "sample/__init__.py": "",
            "sample/fine.py": "",
            "sample/needs.py": "import a_dependency_that_is_not_installed\n",
            "sample/broken.py": "raise ValueError('at import')\n",
            "sample/__main__.py": "raise SystemExit('a program, not a module')\n",
            "sample/sub/__init__.py": "import sys\nsys.exit(3)\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    outcomes = bundle_gate.import_package("sample")

    assert outcomes == {
        "sample": "ok",
        "sample.fine": "ok",
        "sample.needs": "ModuleNotFoundError: a_dependency_that_is_not_installed",
        "sample.broken": "ValueError",
        "sample.sub": "SystemExit",
    }


def test_a_module_that_imports_differently_from_the_zip_fails_the_gate(
    tmp_path: Path, monkeypatch
):
    runtime = tmp_path / "runtime"
    site = write(runtime / "site", {"pure/__init__.py": "", "pure/mod.py": ""})
    site_zip.pack(runtime, Path(sys.executable))
    record = tmp_path / "record.json"
    record.write_text(
        '{"pure": "ok", "pure.mod": "ok", "pure.gone": "ok"}', encoding="utf-8"
    )
    monkeypatch.setattr(
        bundle_gate,
        "import_packages",
        lambda runtime, packages: {"pure": "ok", "pure.mod": "ImportError"},
    )

    problems = bundle_gate.zipped_imports(runtime, record)

    assert len(problems) == 2
    assert (
        "pure.gone imported as a file with `ok`" in problems[0]
        and "is not there" in problems[0]
    )
    assert (
        "pure.mod imported as a file with `ok` and from its zip with `ImportError`"
        in problems[1]
    )
    assert site.exists()


def test_a_module_missing_another_dependency_than_before_has_changed():
    """By type alone, 96 modules that fail for want of an optional
    dependency would hide one that now fails for want of something else."""
    before = bundle_gate.outcome(
        ModuleNotFoundError("No module named 'tensorflow'", name="tensorflow")
    )
    after = bundle_gate.outcome(
        ModuleNotFoundError("No module named 'pure.data'", name="pure.data")
    )
    assert (before, after) == (
        "ModuleNotFoundError: tensorflow",
        "ModuleNotFoundError: pure.data",
    )
    assert bundle_gate.outcome(None) == "ok"
    assert bundle_gate.outcome(ValueError("C:\\a path\\that differs")) == "ValueError"


def test_a_zipped_package_that_does_not_import_at_all_fails_the_gate(
    tmp_path: Path, monkeypatch
):
    """None of its modules is tried then, before or after: moviepy's 78 were
    outside the comparison while the gate did not have a service's
    environment, and it said `0 import differently`."""
    runtime = tmp_path / "runtime"
    write(
        runtime / "site",
        {"pure/__init__.py": "", "moviepy/__init__.py": "", "moviepy/video.py": ""},
    )
    site_zip.pack(runtime, Path(sys.executable))
    same = {"pure": "ok", "moviepy": "RuntimeError"}
    record = tmp_path / "record.json"
    record.write_text('{"pure": "ok", "moviepy": "RuntimeError"}', encoding="utf-8")
    monkeypatch.setattr(bundle_gate, "import_packages", lambda runtime, packages: same)

    problems = bundle_gate.zipped_imports(runtime, record)

    assert len(problems) == 1
    assert "the zipped package moviepy does not import (RuntimeError)" in problems[0]

    monkeypatch.setitem(bundle_gate.MAY_NOT_IMPORT, "moviepy", "looked at by hand")
    assert bundle_gate.zipped_imports(runtime, record) == []
    monkeypatch.setitem(bundle_gate.MAY_NOT_IMPORT, "pure", "no longer true")
    assert "pure imports now" in bundle_gate.zipped_imports(runtime, record)[0]


def test_the_gate_imports_a_package_in_the_environment_a_service_has():
    """imageio_ffmpeg, and moviepy with it, imports only where it is told
    where ffmpeg is; a service is (settings.py)."""
    gate = (DESKTOP / "build" / "bundle_gate.py").read_text(encoding="utf-8")
    child = gate[gate.index("if args.package:") : gate.index("if args.in_the_backend:")]
    assert "as_a_service(runtime" in child and child.index(
        "as_a_service"
    ) < child.index("import_package")
    assert (
        bundle_gate.MAY_NOT_IMPORT == {}
    ), "each entry is a package whose modules nothing compares"


def test_the_backend_modules_that_may_fail_to_import_are_still_there():
    for module in bundle_gate.NOT_IMPORTABLE:
        path = BACKEND.parent / (module.replace(".", "/") + ".py")
        assert path.is_file(), (
            f"{module} is gone from the backend: take it off NOT_IMPORTABLE in "
            "desktop/build/bundle_gate.py"
        )
