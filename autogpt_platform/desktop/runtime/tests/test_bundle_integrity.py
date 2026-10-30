"""The installed bundle is never written to, and brings its own tools.

Three kinds of test: what the runtime hands the services so that they write
elsewhere and find ffmpeg; the build helpers that edit the Next server's
configuration and place ffmpeg, against fixtures; and the upstream facts all
of it rests on, each failing with what to re-read when upstream moves.
"""

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from autogpt_desktop import ports, rabbitmq, settings
from autogpt_desktop.layout import EXE, Bundle, DataDir

DESKTOP = Path(__file__).resolve().parents[2]
PLATFORM = DESKTOP.parent
# The build's modules are a script directory's, not a package's.
sys.path.insert(0, str(DESKTOP / "build"))
lock_export = importlib.import_module("lock_export")
next_config = importlib.import_module("next_config")


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return data


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path / "runtime")


def backend_env(
    bundle: Bundle, data: DataDir, user: dict[str, str] | None = None
) -> dict[str, str]:
    secret = settings.ensure_secrets(bundle, data)
    return settings.backend_environment(
        bundle, data, ports.allocate(data.ports_file), secret, user or {}
    )


def with_ffmpeg(bundle: Bundle) -> Bundle:
    bundle.tools_bin.mkdir(parents=True)
    bundle.ffmpeg.write_bytes(b"")
    return bundle


# --- what the services are told ------------------------------------------------


def test_the_backends_file_logging_cannot_land_in_the_bundle(
    bundle: Bundle, data: DataDir
):
    """Off by default; settings.env can turn it on, but not choose the bundle."""
    assert backend_env(bundle, data)["LOG_DIR"] == str(data.logs / "backend")
    user = {"ENABLE_FILE_LOGGING": "true", "LOG_DIR": str(bundle.root / "logs")}
    assert backend_env(bundle, data, user)["LOG_DIR"] == str(data.logs / "backend")


def test_log_dir_is_still_how_the_backend_is_told_where_to_log():
    config = (
        PLATFORM / "autogpt_libs" / "autogpt_libs" / "logging" / "config.py"
    ).read_text("utf-8")
    assert (
        re.search(r"^\s+log_dir: Path = Field\(", config, re.M)
        and 'env_prefix=""' in config
    ), (
        "autogpt_libs/logging/config.py no longer reads its log directory from LOG_DIR. Its "
        "default is a directory beside the code, inside the bundle: find the new setting and "
        "set it in settings.backend_environment."
    )


def test_services_find_the_bundled_ffmpeg_by_name_and_through_imageio(
    bundle: Bundle, data: DataDir, monkeypatch
):
    monkeypatch.setenv("PATH", "/usr/local/bin")
    env = backend_env(with_ffmpeg(bundle), data, {"PATH": "/from/settings.env"})

    assert env["PATH"].split(settings.os.pathsep) == [
        str(bundle.tools_bin),
        "/usr/local/bin",
    ]
    assert env["IMAGEIO_FFMPEG_EXE"] == str(
        bundle.root / "tools" / "bin" / f"ffmpeg{EXE}"
    )


def test_a_bundle_without_ffmpeg_leaves_the_search_to_the_packages(
    bundle: Bundle, data: DataDir, caplog
):
    """A variable naming a file that is not there would stop imageio-ffmpeg
    looking anywhere else."""
    env = backend_env(bundle, data)
    assert "IMAGEIO_FFMPEG_EXE" not in env and "PATH" not in env
    assert "has no ffmpeg" in caplog.text


def test_the_wheel_that_carries_ffmpeg_is_still_locked_on_every_system():
    for platform in ("win32", "darwin", "linux"):
        lines = lock_export.export(PLATFORM / "backend" / "poetry.lock", platform)
        assert any(line.startswith("imageio-ffmpeg==") for line in lines), (
            f"the backend's lock no longer installs imageio-ffmpeg on {platform}. The bundle's "
            "ffmpeg is the binary inside that wheel (desktop/build/bundled_tools.py): bundle "
            "one from somewhere else, pinned."
        )


def test_nothing_is_ever_created_inside_the_bundle_for_rabbitmq(
    bundle: Bundle, data: DataDir, monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    port = {"epmd": 15001, "rabbitmq_dist": 15002}
    try:
        env = rabbitmq.environment(bundle, data, port)
    except (RuntimeError, OSError):
        env = None  # no alias to a bundle that is not there; that is not the point
    assert not bundle.root.exists()
    if env:
        assert Path(env["RABBITMQ_LOG_BASE"]).parent == Path(env["RABBITMQ_BASE"])
        assert not env["RABBITMQ_LOG_BASE"].startswith(env["RABBITMQ_HOME"])


def test_the_data_directory_has_no_place_for_a_backend_config():
    """backend/config.json is neither written nor read (Settings.save has no
    caller), so there is nothing to redirect and nowhere kept for it."""
    assert not hasattr(DataDir, "backend_config")


def test_the_backend_files_that_resolve_paths_beside_the_code_are_the_known_ones():
    """get_data_path() is the backend package's parent directory: in the
    bundle. The two live users follow WORKSPACE_STORAGE_DIR, which the runtime
    sets; settings.py uses it only for the config.json nobody saves."""
    backend = PLATFORM / "backend" / "backend"
    callers = {
        path.relative_to(backend).as_posix()
        for path in backend.rglob("*.py")
        if not path.name.endswith("_test.py")
        and "get_data_path(" in path.read_text("utf-8")
    }
    assert callers == {
        "util/data.py",
        "util/settings.py",
        "util/workspace_storage_local.py",
        "api/features/store/local_media.py",
    }, (
        f"the backend files calling get_data_path() are now {sorted(callers)}. A new caller "
        "may write beside the code, which is inside the installed bundle: see where it writes "
        "and give it a directory in the data directory (settings.backend_environment), then "
        "update this list."
    )


# --- the Next server's embedded configuration ----------------------------------

CONFIG = {
    "env": {},
    "distDir": "./.next",
    "images": {"deviceSizes": [640], "minimumCacheTTL": 60, "path": "/_next/image"},
    "experimental": {"isrFlushToDisk": True, "cpus": 7},
    "htmlLimitedBots": "Mediapartners-Google|Slurp\\b|bingbot",
    "turbopack": {"root": "C:\\Users\\nicka\\code\\frontend"},
    "name": "caf\u00e9",
}


def server_js(config: dict) -> str:
    embedded = json.dumps(config, separators=(",", ":"), ensure_ascii=False)
    return (
        "const path = require('path')\n\nprocess.chdir(__dirname)\n"
        f"const nextConfig = {embedded}\n\n"
        "process.env.__NEXT_PRIVATE_STANDALONE_CONFIG = JSON.stringify(nextConfig)\n"
    )


def frontend(tmp_path: Path, config: dict = CONFIG, newline: str = "\n") -> Path:
    root = tmp_path / "frontend"
    (root / ".next").mkdir(parents=True)
    with open(root / "server.js", "w", encoding="utf-8", newline="") as stream:
        stream.write(server_js(config).replace("\n", newline))
    required = {"version": 1, "config": config, "files": [".next/BUILD_ID"]}
    with open(
        root / ".next" / "required-server-files.json", "w", encoding="utf-8", newline=""
    ) as stream:
        stream.write(json.dumps(required, indent=2, ensure_ascii=False))
    return root


def embedded(root: Path) -> dict:
    with open(root / "server.js", encoding="utf-8", newline="") as stream:
        (found,) = next_config.EMBEDDED_CONFIG.findall(stream.read())
    return json.loads(found)


def test_both_copies_of_the_configuration_get_the_three_settings(tmp_path: Path):
    root = frontend(tmp_path)

    next_config.keep_caches_off_disk(root)

    required = json.loads(
        (root / ".next" / "required-server-files.json").read_text("utf-8")
    )
    for config in (embedded(root), required["config"]):
        assert config["experimental"] == {"isrFlushToDisk": False, "cpus": 7}
        assert config["images"]["maximumDiskCacheSize"] == 0
        assert config["images"]["minimumCacheTTL"] == 14400
        untouched = {
            key: value
            for key, value in config.items()
            if key not in ("experimental", "images")
        }
        assert untouched == {
            k: v for k, v in CONFIG.items() if k not in ("experimental", "images")
        }
    assert required["files"] == [".next/BUILD_ID"]
    next_config.check(root)


def test_nothing_but_the_configuration_changes_in_server_js(tmp_path: Path):
    """Backslashes, non-ASCII text and line ends come back byte for byte."""
    root = frontend(tmp_path)
    before = (root / "server.js").read_bytes()

    next_config.keep_caches_off_disk(root)
    after = (root / "server.js").read_bytes()

    wanted = json.loads(json.dumps(CONFIG))
    wanted["experimental"]["isrFlushToDisk"] = False
    wanted["images"].update(maximumDiskCacheSize=0, minimumCacheTTL=14400)
    assert after == server_js(wanted).encode("utf-8")
    assert b"\r" not in after
    assert before.split(b"\n")[:3] == after.split(b"\n")[:3]

    next_config.keep_caches_off_disk(root)  # applying it again changes nothing
    assert (root / "server.js").read_bytes() == after


def test_a_build_whose_configuration_is_not_where_expected_fails(tmp_path: Path):
    root = frontend(tmp_path)
    text = (root / "server.js").read_text("utf-8")

    (root / "server.js").write_text(
        text.replace("const nextConfig", "let config"), "utf-8"
    )
    with pytest.raises(next_config.NextConfigError, match="found 0"):
        next_config.keep_caches_off_disk(root)

    twice = text + "const nextConfig = {}\n"
    with open(root / "server.js", "w", encoding="utf-8", newline="") as stream:
        stream.write(twice)
    with pytest.raises(next_config.NextConfigError, match="found 2"):
        next_config.keep_caches_off_disk(root)


def test_a_server_js_with_windows_line_ends_is_refused_not_corrupted(tmp_path: Path):
    root = frontend(tmp_path, newline="\r\n")
    before = (root / "server.js").read_bytes()
    with pytest.raises(next_config.NextConfigError):
        next_config.keep_caches_off_disk(root)
    assert (root / "server.js").read_bytes() == before


@pytest.mark.parametrize(
    ("section", "key"),
    [("experimental", "isrFlushToDisk"), ("images", "minimumCacheTTL")],
)
def test_a_next_that_dropped_a_setting_fails_the_build_instead_of_writing_again(
    tmp_path: Path, section: str, key: str
):
    config = json.loads(json.dumps(CONFIG))
    del config[section][key]
    root = frontend(tmp_path, config)
    before = (root / "server.js").read_bytes()

    with pytest.raises(next_config.NextConfigError, match=key):
        next_config.keep_caches_off_disk(root)
    assert (root / "server.js").read_bytes() == before


def test_a_frontend_that_was_not_rewritten_does_not_pass_the_check(tmp_path: Path):
    with pytest.raises(next_config.NextConfigError, match="isrFlushToDisk"):
        next_config.check(frontend(tmp_path))


def test_the_frontend_still_builds_a_standalone_server():
    config = (PLATFORM / "frontend" / "next.config.mjs").read_text("utf-8")
    assert re.search(r'output:\s*"standalone"', config), (
        "frontend/next.config.mjs no longer builds a standalone server; the desktop "
        "bundle ships that output and edits its embedded configuration "
        "(desktop/build/next_config.py, build_runtime.step_frontend)."
    )
