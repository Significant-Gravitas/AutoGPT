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
    # The entry points of the application
    executables=[
        Executable(
            "autogpt_server/app.py",
            target_name="agpt_server",
            base="console",
            icon=icon,
        ),
        Executable(
            "autogpt_server/cli.py",
            target_name="agpt_server_cli",
            base="console",
            icon=icon,
        ),
    ],
    options={
        # Options for building all the executables
        "build_exe": {
            "packages": packages,
            "includes": [
                "autogpt_server",
                "uvicorn.loops.auto",
                "uvicorn.protocols.http.auto",
                "uvicorn.protocols.websockets.auto",
                "uvicorn.lifespan.on",
            ],
            # Exclude the two module from readability.compat as it causes issues
            "excludes": ["readability.compat.two"],
        },
        # Mac .app specific options
        "bdist_mac": {
            "bundle_name": "AutoGPT",
            "iconfile": "../../assets/gpt_dark_RGB.icns",
            # "include_resources": ["IMG_3775.jpeg"],
        },
        # Mac .dmg specific options
        "bdist_dmg": {
            "applications_shortcut": True,
            "volume_label": "AutoGPTServer",
        },
        # Windows .msi specific options
        "bdist_msi": {
            "target_name": "AutoGPTServer",
            "add_to_path": True,
            "install_icon": "../../assets/gpt_dark_RGB.ico",
        },
        # Linux .appimage specific options
        "bdist_appimage": {},
        # Linux rpm specific options
        "bdist_rpm": {
            "name": "AutoGPTServer",
            "description": "AutoGPT Server",
            "version": "0.1",
            "license": "UNKNOWNORPROPRIETARY",
            "url": "https://agpt.co",
            "long_description": "AutoGPT Server",
        },
    },
)
