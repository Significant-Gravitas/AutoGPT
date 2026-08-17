import os
import subprocess
import sys
from collections.abc import Mapping

_PASSTHROUGH = frozenset(
    {
        "APPDATA",
        "CODEX_HOME",
        "CODEX_SQLITE_HOME",
        "COMSPEC",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "NO_COLOR",
        "NO_PROXY",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "RUST_LOG",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)


def sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    ambient = os.environ if source is None else source
    child = {
        key: value for key, value in ambient.items() if key.upper() in _PASSTHROUGH
    }
    child.update(
        {
            "CODEX_ACCESS_TOKEN": "",
            "CODEX_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
    )
    return child


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Codex launcher requires the runtime path")
    runtime, *arguments = sys.argv[1:]
    if os.name == "nt":
        completed = subprocess.run(
            [runtime, *arguments],
            env=sanitized_environment(),
            check=False,
        )
        raise SystemExit(completed.returncode)
    os.execve(runtime, [runtime, *arguments], sanitized_environment())


if __name__ == "__main__":
    main()
