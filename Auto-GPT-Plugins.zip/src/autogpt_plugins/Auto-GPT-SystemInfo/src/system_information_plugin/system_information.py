"""Gets system information."""
from __future__ import annotations

import os
import platform

import distro


def get_system_information() -> str:
    """
    Gets system information.

    Returns:
        str: The system information.
    """

    # Get system architecture
    arch = platform.architecture()[0]

    # Get system distribution (works on Linux only)
    if platform.system() == "Linux":
        distro_name = distro.name()
        distro_version = distro.version()
        distro_id = distro.id()

        distro_info = f"{distro_name} {distro_version} ({distro_id})"
    else:
        distro_info = None

    # Get Windows version (works on Windows only)
    if platform.system() == "Windows":
        win_ver = platform.win32_ver()
        win_version = f"{win_ver[0]} {win_ver[1]} {win_ver[3]}"
    else:
        win_version = None

    # Get macOS version (works on macOS only)
    if platform.system() == "Darwin":
        mac_ver = platform.mac_ver()
        mac_version = f"{mac_ver[0]} {mac_ver[2]}"
    else:
        mac_version = None

    # Build the prompt
    if distro_info:
        os_info = f"Linux {arch} {distro_info}"
    if win_version:
        os_info = f"Windows {win_version}"
    if mac_version:
        os_info = f"macOS {mac_version}"

    return os_info


def get_shell_name() -> str:
    """Gets the shell name.

    Returns:
        str: The shell name.

    Throws:
        KeyError: If the shell name is not found.
        AttributeError: If the shell name is not found.
    """

    try:
        if platform.system() == "Windows":
            shell = os.environ.get("ComSpec", "").split("\\")[-1]
            if shell.lower() == "cmd.exe":
                return "cmd"
            elif shell.lower() == "powershell.exe":
                return "powershell"
        else:
            shell = os.environ.get("SHELL", "").split("/")[-1]
            if shell in ["bash", "zsh", "sh", "ksh", "csh", "fish", "dash"]:
                return shell
    except (KeyError, AttributeError) as e:
        print(f"Error: {e}")

    return ""
