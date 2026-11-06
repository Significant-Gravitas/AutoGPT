"""Private CLI directory creation, without changing preexisting permissions."""

import os
import stat
from contextlib import ExitStack

if os.name == "nt":
    from backend.copilot import windows_private


def ensure_private_directory(path: str, *, lifetime: ExitStack | None = None) -> str:
    if os.name == "nt":
        return windows_private.ensure_private_directory(path, lifetime=lifetime)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        st = os.lstat(path)
        if (
            not stat.S_ISDIR(st.st_mode)
            or st.st_uid != os.getuid()
            or st.st_mode & 0o077
        ):
            raise
    return path
