from pkgutil import iter_modules
from shutil import which

from cx_Freeze import Executable, setup

packages = [
    m.name
    for m in iter_modules()
    if m.ispkg and m.module_finder and "poetry" in m.module_finder.path  # type: ignore
]
packages.append("collections")

# if mac use the icns file, otherwise use the ico file
icon = (
    "../../assets/gpt_dark_RGB.icns"
    if which("sips")
    else "../../assets/gpt_dark_RGB.ico"
)

setup(
    name="AutoGPT Server",
    url="https://agpt.co",
    executables=[
        Executable(
            "autogpt_server/app.py", target_name="server", base="console", icon=icon
        ),
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
        "bdist_msi": {
            "target_name": "AutoGPTServer",
            "add_to_path": True,
            "install_icon": "../../assets/gpt_dark_RGB.ico",
        },
        "bdist_appimage": {},
    },
)
