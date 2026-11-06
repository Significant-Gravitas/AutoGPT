"""Create a fresh Windows data root before the shell writes into it."""

import sys
from pathlib import Path

from . import winlinks


def create(path: Path) -> None:
    if sys.platform != "win32":
        raise OSError("Private Windows root creation requires Windows")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not winlinks.create_private(path, winlinks.current_user_sid()):
        raise FileExistsError("The data folder appeared during private creation")
