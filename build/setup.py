from shutil import which
from pkgutil import iter_modules
from cx_Freeze import setup, Executable

import os

packages = [
    m.name for m in iter_modules() if m.ispkg and ".venv" in m.module_finder.path
]

setup(
    executables=[Executable(which("hypercorn"), target_name="hypercorn")],
    options={
        "build_exe": {
            "packages": packages,
            "includes": ["main"],
        },
    },
)
