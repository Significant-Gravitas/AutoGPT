"""Import and pruning regressions using disposable packages, without a runtime."""

import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUILD))
backend_tests = importlib.import_module("backend_tests")
bundle_gate = importlib.import_module("bundle_gate")


def write(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def probe(path, package, home):
    home.mkdir(exist_ok=True)
    script = (
        "import json, sys\n"
        "def audit(event, args):\n"
        "    if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):\n"
        "        raise RuntimeError('network forbidden in bundle gate regression')\n"
        "sys.addaudithook(audit)\n"
        "sys.path[:0] = sys.argv[1:3]\n"
        "import bundle_gate\n"
        "bundle_gate.as_a_service = lambda *args: None\n"
        "sys.argv = [bundle_gate.__file__, '.', '--package', sys.argv[3]]\n"
        "bundle_gate.main()\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(BUILD),
            str(path),
            package,
        ],
        cwd=home,
        env={
            **{
                key: os.environ[key]
                for key in ("SYSTEMROOT", "PATH")
                if key in os.environ
            },
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TMP": str(home),
            "TEMP": str(home),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def archive(root, target):
    with zipfile.ZipFile(target, "w") as output:
        for path in sorted(root.rglob("*")):
            output.write(path, path.relative_to(root).as_posix())
    return target


def test_test_only_fixtures_are_pruned_but_runtime_fixture_closure_is_kept(tmp_path):
    package = write(
        tmp_path / "backend",
        {
            "__init__.py": "",
            "app.py": "from .runtime_fixtures import VALUE\n",
            "runtime_fixtures.py": "from .nested_fixtures import VALUE\n",
            "nested_fixtures.py": "VALUE = 1\n",
            "recall_test.py": "from .recall_fixtures import VALUE\n",
            "recall_fixtures.py": "import pytest_mock\nVALUE = 1\n",
        },
    )
    backend_tests.prune(package)
    assert not (package / "recall_test.py").exists()
    assert not (package / "recall_fixtures.py").exists()
    assert probe(package.parent, "backend", tmp_path / "home") == {
        "backend": "ok",
        "backend.app": "ok",
        "backend.runtime_fixtures": "ok",
        "backend.nested_fixtures": "ok",
    }


def test_current_backend_recall_fixtures_are_test_only():
    package = BUILD.parents[1] / "backend" / "backend"
    removed = set(backend_tests.removable(package))
    for name in ("cascade", "integration", "marker"):
        assert (
            package / "copilot" / "graphiti" / f"recall_{name}_fixtures.py" in removed
        )


def test_fixture_defining_a_loadable_block_still_stops_pruning(tmp_path):
    package = write(
        tmp_path / "backend",
        {"blocks/example_fixtures.py": "class ExampleBlock(object): pass\n"},
    )
    with pytest.raises(backend_tests.BlockInTestError):
        backend_tests.prune(package)
    assert (package / "blocks/example_fixtures.py").exists()


def test_import_failures_stay_distinct_between_files_and_zip(tmp_path):
    site = write(
        tmp_path / "site",
        {
            "sample/__init__.py": "",
            "sample/ok.py": "",
            "sample/optional.py": "import gate_missing_optional\n",
            "sample/wrapped.py": (
                "try:\n    import gate_missing_sdk\n"
                "except ImportError:\n    raise ImportError('install the SDK')\n"
            ),
            "sample/__main__.py": "raise SystemExit('must not execute')\n",
        },
    )
    before = probe(site, "sample", tmp_path / "before")
    after = probe(archive(site, tmp_path / "site.zip"), "sample", tmp_path / "after")
    assert before == after
    assert before["sample.optional"] == "ModuleNotFoundError: gate_missing_optional"
    assert before["sample.wrapped"] != "ok"
    assert "sample.__main__" not in before


def test_import_probe_does_not_install_optional_dependencies(tmp_path):
    installer = write(
        tmp_path / "installer",
        {
            "pip/__init__.py": "",
            "pip/__main__.py": (
                "from pathlib import Path\n"
                "Path('installer-ran').write_text('bundle mutated')\n"
            ),
        },
    )
    site = write(
        tmp_path / "site",
        {
            "sample/__init__.py": "",
            "sample/auto_install.py": (
                "import subprocess, sys\n"
                "try:\n    import gate_missing_litellm\n"
                "except ImportError:\n"
                "    subprocess.check_call([sys.executable, '-S', '-B', '-m', 'pip', "
                f"'install', 'litellm'], cwd={str(installer)!r})\n"
            ),
        },
    )
    before = probe(site, "sample", tmp_path / "before")
    assert not (installer / "installer-ran").exists()
    after = probe(archive(site, tmp_path / "site.zip"), "sample", tmp_path / "after")
    assert not (installer / "installer-ran").exists()
    assert before == after
    assert before["sample"] == "ok"
    assert before["sample.auto_install"].startswith("ImportInstallBlocked <- ")
    assert "gate_missing_litellm" in before["sample.auto_install"]
    assert before["sample.auto_install"] != "ok"


def test_import_probe_preserves_non_installer_process_checks(tmp_path):
    site = write(
        tmp_path / "site",
        {
            "sample/__init__.py": (
                "import subprocess, sys\n"
                "subprocess.run([sys.executable, '-I', '-S', '-B', '-c', 'pass'], "
                "check=True, timeout=5)\n"
            ),
        },
    )
    assert probe(site, "sample", tmp_path / "home") == {"sample": "ok"}


@pytest.mark.parametrize("explicit_cause", [False, True])
def test_wrapped_imports_of_a_different_missing_dependency_fail_parity(
    tmp_path, monkeypatch, explicit_cause
):
    cause = " from exc" if explicit_cause else ""
    site = write(
        tmp_path / "site",
        {
            "sample/__init__.py": "",
            "sample/optional.py": (
                "try:\n    import gate_optional_sdk\n"
                f"except ImportError as exc:\n    raise ImportError('optional SDK'){cause}\n"
            ),
        },
    )
    record = tmp_path / "before.json"
    record.write_text(json.dumps(probe(site, "sample", tmp_path / "before")), "utf-8")
    write(
        site,
        {
            "sample/optional.py": (
                "try:\n    import sample.missing_data_module\n"
                f"except ImportError as exc:\n    raise ImportError('optional SDK'){cause}\n"
            )
        },
    )
    archives = tmp_path / "site-zip"
    archives.mkdir()
    packed = archive(site, archives / "00.zip")
    monkeypatch.setattr(
        bundle_gate,
        "import_packages",
        lambda runtime, packages: probe(packed, "sample", tmp_path / "after"),
    )
    problems = bundle_gate.zipped_imports(tmp_path, record)
    assert len(problems) == 1
    assert "sample.optional" in problems[0]
    assert "gate_optional_sdk" in problems[0]
    assert "sample.missing_data_module" in problems[0]


@pytest.mark.parametrize("lost", ["openai/pagination.py", "openai"])
def test_a_missing_openai_module_or_package_still_fails_parity(
    tmp_path, monkeypatch, lost
):
    site = write(
        tmp_path / "site",
        {"openai/__init__.py": "", "openai/pagination.py": "", "other/__init__.py": ""},
    )
    record = tmp_path / "before.json"
    before = probe(site, "openai", tmp_path / "before")
    before.update(probe(site, "other", tmp_path / "other"))
    record.write_text(json.dumps(before), "utf-8")
    archives = tmp_path / "site-zip"
    archives.mkdir()
    packed = archives / "00.zip"
    with zipfile.ZipFile(packed, "w") as output:
        for path in sorted(site.rglob("*")):
            name = path.relative_to(site).as_posix()
            if name != lost and not name.startswith(lost + "/"):
                output.write(path, name)

    def check_packages(runtime, packages):
        result = {}
        for package in packages:
            result.update(probe(packed, package, tmp_path / f"after-{package}"))
        return result

    monkeypatch.setattr(bundle_gate, "import_packages", check_packages)
    problems = bundle_gate.zipped_imports(tmp_path, record)
    assert any(
        "openai.pagination" in problem and "is not there" in problem
        for problem in problems
    )
    if lost == "openai":
        assert any(
            "openai imported" in problem and "is not there" in problem
            for problem in problems
        )
