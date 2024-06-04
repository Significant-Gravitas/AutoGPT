from shutil import which
from pkgutil import iter_modules

from cx_Freeze import setup, Executable

import os

packages = [
    m.name
    for m in iter_modules()
    if m.ispkg and (".venv" in m.module_finder.path or "poetry" in m.module_finder.path)
]
packages.append("collections")

setup(
    name="AutoGPT Server",
    url="https://agpt.co",
    executables=[
        Executable("autogpt_server/app.py", target_name="server"),
    ],
    options={
        "build_exe": {
            "packages": packages,
            "includes": [
                "autogpt_server",
                "uvicorn.loops.auto",
                "uvicorn.protocols.http.auto",
                "uvicorn.protocols.websockets.auto",
                "uvicorn.lifespan.on",
            ],
            "excludes": ["readability.compat.two"],
        },
        "bdist_mac": {
            "bundle_name": "AutoGPT",
            "iconfile": "../../assets/gpt_dark_RGB.icns",
            # "include_resources": ["IMG_3775.jpeg"],
        },
        "bdist_dmg": {
            "applications_shortcut": True,
            "volume_label": "AutoGPT Server",
        },
        "bdist_msi": {},
        "bdist_appimage": {},
    },
)
