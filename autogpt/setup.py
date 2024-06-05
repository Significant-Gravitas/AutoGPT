from pkgutil import iter_modules
from shutil import which

from cx_Freeze import Executable, setup

packages = [
    m.name
    for m in iter_modules()
    if m.ispkg
    and m.module_finder
    and ("poetry" in m.module_finder.path)  # type: ignore
]

icon = (
    "../../assets/gpt_dark_RGB.icns"
    if which("sips")
    else "../../assets/gpt_dark_RGB.ico"
)


setup(
    executables=[
        Executable(
            "autogpt/__main__.py", target_name="autogpt", base="console", icon=icon
        ),
    ],
    options={
        "build_exe": {
            "packages": packages,
            "includes": [
                "autogpt",
                "spacy",
                "spacy.lang",
                "spacy.vocab",
                "spacy.lang.lex_attrs",
                "uvicorn.loops.auto",
                "srsly.msgpack.util",
                "blis",
                "uvicorn.protocols.http.auto",
                "uvicorn.protocols.websockets.auto",
                "uvicorn.lifespan.on",
            ],
            "excludes": ["readability.compat.two"],
        },
        "bdist_mac": {
            "bundle_name": "AutoGPT",
            "iconfile": "../assets/gpt_dark_RGB.icns",
            "include_resources": [""],
        },
        "bdist_dmg": {
            "applications_shortcut": True,
            "volume_label": "AutoGPT",
        },
        "bdist_msi": {
            "target_name": "AutoGPT",
            "add_to_path": True,
            "install_icon": "../assets/gpt_dark_RGB.ico",
        },
    },
)
