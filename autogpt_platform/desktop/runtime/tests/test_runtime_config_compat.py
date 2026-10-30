"""Runtime configuration compatibility before the shared Windows fix lands."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogpt_desktop import settings
from autogpt_desktop.layout import Bundle, DataDir

RUNTIME_CONFIG = "autogpt_platform/single-container/runtime_config.py"
PATCH = "autogpt_platform/desktop/upstream/runtime-config-windows.patch"


# Every name `os` has on Windows: CPython 3.13.15, `sorted(dir(os))` without
# the dunder names. test_the_recorded_windows_os_is_the_real_one holds it true.
WINDOWS_OS_PYTHON = (3, 13)
WINDOWS_OS_NAMES = """
    DirEntry EX_OK F_OK GenericAlias Mapping MutableMapping O_APPEND O_BINARY O_CREAT O_EXCL
    O_NOINHERIT O_RANDOM O_RDONLY O_RDWR O_SEQUENTIAL O_SHORT_LIVED O_TEMPORARY O_TEXT O_TRUNC
    O_WRONLY P_DETACH P_NOWAIT P_NOWAITO P_OVERLAY P_WAIT PathLike R_OK SEEK_CUR SEEK_END
    SEEK_SET TMP_MAX W_OK X_OK _AddedDllDirectory _Environ _check_methods _execvpe _exists
    _exit _fspath _get_exports_list _walk_symlinks_as_files _wrap_close abc abort access
    add_dll_directory altsep chdir chmod close closerange cpu_count curdir defpath
    device_encoding devnull dup dup2 environ error execl execle execlp execlpe execv execve
    execvp execvpe extsep fchmod fdopen fsdecode fsencode fspath fstat fsync ftruncate
    get_blocking get_exec_path get_handle_inheritable get_inheritable get_terminal_size getcwd
    getcwdb getenv getlogin getpid getppid isatty kill lchmod linesep link listdir listdrives
    listmounts listvolumes lseek lstat makedirs mkdir name open pardir path pathsep pipe popen
    process_cpu_count putenv read readlink remove removedirs rename renames replace rmdir
    scandir sep set_blocking set_handle_inheritable set_inheritable spawnl spawnle spawnv
    spawnve st startfile stat stat_result statvfs_result strerror supports_bytes_environ
    supports_dir_fd supports_effective_ids supports_fd supports_follow_symlinks symlink sys
    system terminal_size times times_result truncate umask uname_result unlink unsetenv
    urandom utime waitpid waitstatus_to_exitcode walk write
"""
WINDOWS_OS = frozenset(WINDOWS_OS_NAMES.split())
# Windows has had it only since Python 3.13; the app must not need it.
NEWER_THAN_SUPPORTED = {"fchmod"}


class LikeWindows:
    """`os` as Windows has it, on any system: nothing Windows lacks, no
    fchmod (new there in Python 3.13), and directories that cannot be
    opened. What is left is this system's own."""

    name = "nt"

    def __getattr__(self, name: str):
        if name not in WINDOWS_OS or name in NEWER_THAN_SUPPORTED:
            raise AttributeError(
                f"{RUNTIME_CONFIG} uses os.{name}, which Windows does not have. Teach "
                "WindowsOs in desktop/runtime/autogpt_desktop/settings.py to stand in for it, "
                f"and add the guard to {PATCH}."
            )
        return getattr(os, name)

    def open(self, path, flags: int, mode: int = 0o777) -> int:
        if os.path.isdir(path):
            raise PermissionError(13, "Permission denied", str(path))
        return os.open(path, flags, mode)


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return data


def test_upstreams_runtime_config_runs_unmodified_on_this_system(
    tmp_path: Path, data: DataDir
):
    """The real module, the real OS: Windows, macOS and Linux each run this."""
    bundle = Bundle(
        tmp_path / "no-bundle"
    )  # so the source tree's copy is the one loaded
    first = settings.ensure_secrets(bundle, data)
    assert first == settings.ensure_secrets(bundle, data)
    assert data.runtime_env.read_text(encoding="ascii").count("\n") > 5
    leftovers = [
        path.name for path in data.config.iterdir() if path.name.endswith(".tmp")
    ]
    assert not leftovers


def test_upstreams_runtime_config_runs_where_windows_refuses(tmp_path: Path):
    """The same module with Windows's `os`, on any system: this is the one
    the upstream sync runs, on Linux. Anything upstream starts to use that
    Windows lacks stops the sync here, not a Windows build afterwards."""
    module = settings.load_runtime_config_module(Bundle(tmp_path))
    assert vars(module).get("os") is not None, (
        "single-container/runtime_config.py no longer does `import os`, so "
        "settings.WindowsOs has nothing to stand in for. See how it reaches fchmod and the "
        "directory fsync now, and update load_runtime_config_module."
    )
    vars(module)["os"] = settings.WindowsOs(LikeWindows())
    path = tmp_path / "config" / "runtime.env"
    first = module.ensure_runtime_config(path, {})
    assert first == module.ensure_runtime_config(path, {})
    assert [entry.name for entry in path.parent.iterdir()] == ["runtime.env"]


def test_the_stand_in_refuses_what_windows_refuses(tmp_path: Path):
    windows = LikeWindows()
    for missing in (
        "fchmod",
        "fchown",
        "geteuid",
        "getuid",
        "chown",
        "O_NOFOLLOW",
        "O_CLOEXEC",
    ):
        assert not hasattr(windows, missing), missing
    with pytest.raises(
        AttributeError, match=r"uses os\.fchown, which Windows does not have"
    ):
        windows.fchown(0, 0, 0)
    with pytest.raises(PermissionError):
        windows.open(tmp_path, os.O_RDONLY)
    assert windows.name == "nt" and windows.sep == os.sep


def test_the_recorded_windows_os_is_the_real_one():
    if sys.platform != "win32":
        pytest.skip("only Windows can say what its `os` has")
    if sys.version_info[:2] < WINDOWS_OS_PYTHON:
        pytest.skip(
            f"recorded from Python {WINDOWS_OS_PYTHON[0]}.{WINDOWS_OS_PYTHON[1]}"
        )
    real = {name for name in dir(os) if not name.startswith("__")}
    assert real == WINDOWS_OS and sys.version_info[:2] == WINDOWS_OS_PYTHON, (
        f"`os` on Windows Python {sys.version_info[0]}.{sys.version_info[1]} has "
        f"{sorted(real - WINDOWS_OS)} more and {sorted(WINDOWS_OS - real)} fewer names than "
        "WINDOWS_OS records. Record them (the comment above it says how) with WINDOWS_OS_PYTHON."
    )


def test_windows_os_leaves_files_alone_and_skips_only_the_directory(tmp_path: Path):
    synced: list[int] = []
    closed: list[int] = []
    real = SimpleNamespace(
        path=os.path,
        devnull=os.devnull,
        O_RDONLY=os.O_RDONLY,
        open=os.open,
        fsync=synced.append,
        close=lambda descriptor: (closed.append(descriptor), os.close(descriptor)),
        sep="/",
    )
    shim = settings.WindowsOs(real)
    file = tmp_path / "file"
    file.write_text("x", encoding="ascii")

    descriptor = shim.open(file, os.O_RDONLY)
    shim.fchmod(descriptor, 0o600)  # the real one has none: nothing to call
    shim.fsync(descriptor)
    shim.close(descriptor)
    assert synced == [descriptor] and closed == [descriptor]

    directory = shim.open(tmp_path, os.O_RDONLY)
    shim.fsync(directory)
    shim.close(directory)
    assert synced == [descriptor], "the directory's stand-in must not be fsynced"
    assert closed == [descriptor, directory], "and must be closed"
    assert shim.sep == "/", "everything else is the real module's"
