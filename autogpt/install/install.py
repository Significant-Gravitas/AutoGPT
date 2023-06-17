"""
This module contains functions for checking if Auto-GPT is installed and prompting the user to install it if not.
"""
import os
from pathlib import Path
import shutil
from typing import Optional

import click
from colorama import Fore
from prompt_toolkit import prompt

def is_autogpt_installed() -> bool:
    """
    Returns True if the autogpt command is installed.

    Returns:
        bool: True if the autogpt command is installed.
    """
    return shutil.which("autogpt") is not None

def get_local_install_directory() -> Optional[Path]:
    """
    Tries to find the local install directory of Auto-GPT, 
    if the user is running Auto-GPT from within the Auto-GPT folder.

    Returns:
        Optional[Path]: The local install directory of Auto-GPT, or None if it couldn't be found.
    """
    if Path("./autogpt").exists():
        return Path("./autogpt")
    elif Path(Path(__file__).parent / "autogpt").exists():
        return Path(Path(__file__).parent / "autogpt")
    else:
        return None
    
def prompt_install_method()->int:
    """
    Prompts the user to select an installation method and returns the corresponding integer value.

    Returns:
        int: The integer value corresponding to the selected installation method.
    """
    print("Select an install method:")
    
    options = []
    if local_install_directory:=get_local_install_directory():
        options.append(f"Local directory: {local_install_directory} - `pip install .`")
    options.append("PyPI - `pip install autogpt`")
    options.append("TestPyPI - `pip install --index-url https://test.pypi.org/simple/ autogpt`")
    options.append("Skip installation.")
    print("\n".join([f"{i+1}. {option}" for i, option in enumerate(options)]))
    return int(prompt("Enter the number of your choice: "))

def install_autogpt() -> None:
    """
    Prompts the user to select an installation method and kicks off the installation using the selected method.
    """
    method = prompt_install_method()
    if method == 1:
        cwd = Path.cwd()
        os.chdir(get_local_install_directory().parent)
        os.system("pip install .")
        os.chdir(cwd)
    elif method == 2:
        os.system("pip install autogpt")
    elif method == 3:
        os.system("pip install --index-url https://test.pypi.org/simple/ agpt")
    elif method == 4:
        print(
            Fore.YELLOW
            + "You can run 'pip install autogpt' later to install the Auto-GPT command."
            + Fore.RESET
        )

            
def check_installation(workspace_directory: Optional[str], interactive: bool=True)->None:
    """
    Checks if Auto-GPT is being invoked from within the Auto-GPT folder or if the autogpt command doesn't exist
    and prompts the user to install it if so.

    Args:
        workspace_directory (Optional[str]): The workspace directory to use.
        interactive (bool, optional): Whether the user started Auto-GPT in interactive mode or not. Defaults to True.
    """
    if not is_autogpt_installed():
        print(
            Fore.YELLOW
            + "Warning: 'autogpt' command not found. With this command, you can run 'autogpt' from anywhere."
            + Fore.RESET
        )
        if interactive:
            install_autogpt()
        else:
            print(
                Fore.YELLOW
                + "You can run 'pip install autogpt' to install the Auto-GPT command."
                + Fore.RESET
            )