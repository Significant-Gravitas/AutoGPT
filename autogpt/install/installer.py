"""
Installer class for Auto-GPT.
"""
from pathlib import Path
import shutil
import subprocess
from typing import Optional
from platformdirs import user_config_dir
from colorama import Fore
from prompt_toolkit import prompt
from autogpt.logs import logger

PREFS_FILE = ".install_prefs.txt"
PREFS = [
    "1. pip install agpt",
    "2. pip install -i https://test.pypi.org/simple/ agpt",
    "3. Not now. Ask again next time.",
    "4. Never ask again. Use --install-options to change this.",
]

class Installer:
    def __init__(self):
        """
        Initializes the Installer class.
        """
        prefs_dir = Path(user_config_dir("autogpt"))
        prefs_dir.mkdir(parents=True, exist_ok=True)
        self._prefs_file = Path(prefs_dir, PREFS_FILE)  
        try:
            self._prefs = int(self._prefs_file.read_text()) if self._prefs_file.exists() else None
        except ValueError:
            self._prefs = None
            
        logger.debug(f"Installer initialized with prefs: {self._prefs} [{PREFS[self._prefs-1] if self._prefs else None}]")
        
    def is_installed(self) -> bool:
        """
        Returns True if the autogpt command is installed.

        Returns:
            bool: True if the autogpt command is installed.
        """
        return shutil.which("autogpt") is not None
    
    def run(self, interactive: bool=True)->None:
        """
        Checks if Auto-GPT is being invoked from within the Auto-GPT folder or if the autogpt command doesn't exist
        and prompts the user to install it if so.

        Args:
            workspace_directory (Optional[str]): The workspace directory to use.
            interactive (bool, optional): Whether the user started Auto-GPT in interactive mode or not. Defaults to True.
        """
        if self.is_installed():
            logger.info("Auto-GPT `autogpt` command found. Skipping installation suggestions.")
            return
        
        if not self.is_installed():
            print(
                Fore.YELLOW
                + "AUTO-GPT IS NOT INSTALLED AS A SYSTEM COMMAND.\n"
                + "- 'autogpt' command not found. This is not required, but once installed, you can run 'autogpt' from anywhere."
                + Fore.RESET
            )
            if interactive and self._prefs in (1, 2, 3, None):
                print(
                    Fore.YELLOW
                    + "- You will now be prompted to choose an installation option.\n"
                    + "- This is a One Time Prompt. It will not appear again, unless you select 3 or use `--install-options` to change your selection later.\n"
                    + Fore.RESET
                )
                self._prefs = self.prompt_for_prefs()
                self.save_prefs()
                self.install()
            else:
                print(
                    Fore.YELLOW
                    + "- You can run 'pip install autogpt' to install the Auto-GPT command.\n"
                    + "- You can use --install-options for a guided installation any time.\n"
                    + Fore.RESET
                )
                
                
    def prompt_for_prefs(self) -> int:
        """
        Asks the user for their installation preferences.

        Returns:
            int: The integer value corresponding to the selected installation method.
        """
        print("Choose an installation option:")
        for option in PREFS:
            print(option)
        return int(prompt("Enter the number of your choice: "))

    def install(self) -> None:
        """
        Runs the installation command chosen by the user.
        """
        if self._prefs == 1:
            subprocess.run(["pip", "install", "agpt"])
            print("Successfully installed Auto-GPT using 'pip install agpt'")
        elif self._prefs == 2:
            subprocess.run(["pip", "install", "-i", "https://test.pypi.org/simple/", "agpt"])
            print("Successfully installed Auto-GPT using 'pip install -i https://test.pypi.org/simple/ autogpt'")

    def save_prefs(self) -> None:
        """
        Saves the user's installation preferences to user data.
        """
        logger.debug(f"Saving installation preferences: {self._prefs}")
        self._prefs_file.write_text(str(self._prefs))
        
    def reset_prefs(self) -> None:
        """
        Resets the user's installation preferences.
        """
        logger.debug("Resetting installation preferences.")
        self._prefs_file.unlink(missing_ok=True)
        self._prefs = None