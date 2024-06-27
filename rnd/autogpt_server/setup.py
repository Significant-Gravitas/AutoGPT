import platform
from pathlib import Path
from pkgutil import iter_modules
from typing import Union

from cx_Freeze import Executable, setup # type: ignore

packages = [
    m.name
    for m in iter_modules()
    if m.ispkg and m.module_finder and "poetry" in m.module_finder.path  # type: ignore
]
packages.append("collections")
packages.append("autogpt_server.util.service")
packages.append("autogpt_server.executor.manager")
packages.append("autogpt_server.util.service")

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
txt_to_rtf("../../LICENSE", license_file)


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
                "prisma",
            ],
            # Exclude the two module from readability.compat as it causes issues
            "excludes": ["readability.compat.two"],
            "include_files": [
                # source, destination in the bundle
                # (../frontend, example_files) would also work but you'd need to load the frontend differently in the data.py to correctly get the path when frozen
                ("../example_files", "example_files"),
            ],
        },
        # Mac .app specific options
        "bdist_mac": {
            "bundle_name": "AutoGPT",
            "iconfile": "../../assets/gpt_dark_RGB.icns",
        },
        # Mac .dmg specific options
        "bdist_dmg": {
            "applications_shortcut": True,
            "volume_label": "AutoGPTServer",
            "background": "builtin-arrow",
            
            "license": {
                "default-language": "en_US",
                "licenses": {"en_US": license_file},
                "buttons": {
                    "en_US": [
                        "English",
                        "Agree",
                        "Disagree",
                        "Print",
                        "Save",
                        "If you agree, click Agree to continue the installation. If you do not agree, click Disagree to cancel the installation.",
                    ]
                },
            },
        },
        # Windows .msi specific options
        "bdist_msi": {
            "target_name": "AutoGPTServer",
            "add_to_path": True,
            "install_icon": "../../assets/gpt_dark_RGB.ico",
            "license_file": license_file,
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
