import platform
from pathlib import Path
from pkgutil import iter_modules
from typing import Union

from cx_Freeze import Executable, setup  # type: ignore

packages = [
    m.name
    for m in iter_modules()
    if m.ispkg
    and m.module_finder
    and ("poetry" in m.module_finder.path)  # type: ignore
]

# set the icon based on the platform
icon = "../../assets/gpt_dark_RGB.ico"
if platform.system() == "Darwin":
    icon = "../../assets/gpt_dark_RGB.icns"
elif platform.system() == "Linux":
    icon = "../../assets/gpt_dark_RGB.png"


def txt_to_rtf(input_file: Union[str, Path], output_file: Union[str, Path]) -> None:
    """
    Convert a text file to RTF format.

    Args:
    input_file (Union[str, Path]): Path to the input text file.
    output_file (Union[str, Path]): Path to the output RTF file.

    Returns:
    None
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    with input_path.open("r", encoding="utf-8") as txt_file:
        content = txt_file.read()

    # RTF header
    rtf = r"{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}\f0\fs24 "

    # Replace newlines with RTF newline
    rtf += content.replace("\n", "\\par ")

    # Close RTF document
    rtf += "}"

    with output_path.open("w", encoding="utf-8") as rtf_file:
        rtf_file.write(rtf)


# Convert LICENSE to LICENSE.rtf
license_file = "LICENSE.rtf"
txt_to_rtf("../LICENSE", license_file)


setup(
    executables=[
        Executable(
            "classic/original_autogpt/__main__.py",
            target_name="autogpt",
            base="console",
            icon=icon,
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
            "license_file": license_file,
        },
    },
)
