"""
ContainerConfig class to enable the user to choose between Docker and virtual environment, and start Auto-GPT in the chosen environment.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from platformdirs import user_config_dir
from colorama import Fore
from prompt_toolkit import prompt
from autogpt.logs import logger

PREFS_FILE = ".container_prefs.txt"
PREFS = [
    "1. Always use Docker: Recommended for best security (Requires Docker to be installed.)",
    "2. Always use virtual environment: Avoids dependency conflicts.",
    "3. Run Auto-GPT directly and ask again next time.",
    "4. Run Auto-GPT directly and never ask again. Use --container-options to change this.",
]


class ContainerConfig:
    def __init__(self):
        """
        Initlaizes the ContainerConfig class.
        """
        prefs_dir = Path(user_config_dir("autogpt"))
        prefs_dir.mkdir(parents=True, exist_ok=True)
        self._prefs_file = Path(prefs_dir, PREFS_FILE)
        self._venv_dir = prefs_dir / "venv"
        self._docker_dir = prefs_dir / "docker"

        try:
            self._prefs = (
                int(self._prefs_file.read_text()) if self._prefs_file.exists() else None
            )
        except ValueError:
            self._prefs = None

        logger.debug(
            f"ContainerConfig initialized with prefs: {self._prefs} [{PREFS[self._prefs-1] if self._prefs else None}]"
        )

    def is_docker(self):
        """
        Returns True if Auto-GPT is running in a Docker container.
        """
        return Path("/.dockerenv").exists()

    def is_virtual_env(self):
        """
        Returns True if Auto-GPT is running in a virtual environment.
        """
        return hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

    def run(self, interactive: bool = True) -> None:
        """
        Checks if the user has not selected any container preferences and prompts the user to select one if so.
        Otherwise, it initializes the container based on the user's preferences.

        Args:
            interactive (bool, optional): Whether the user started Auto-GPT in interactive mode or not. Defaults to True.
        """
        # Don't show this to the user if they are running in a docker container.
        if self.is_docker():
            logger.info("Running in Docker container. Skipping container config.")
            # return

        if self._prefs is None or self._prefs == 3:
            print(
                Fore.YELLOW
                + "YOU HAVE NOT SELECTED A CONTAINER OPTION YET.\n"
                + "- It recommended to run Auto-GPT within either a docker container or a virtualenv for increased security and to avoid dependency conflicts.\n"
                + Fore.RESET
            )
            if interactive:
                print(
                    Fore.YELLOW
                    + "This prompt will not appear again, but you can use --container-options to change your selection later.\n"
                    + Fore.RESET
                )
                self._prefs = self.prompt_for_prefs()
                self.save_prefs()
            else:
                print(
                    Fore.YELLOW
                    + "You can use --container-options to make a selection at any time.\n"
                    + Fore.RESET
                )
                if not self.is_docker():
                    print(
                        Fore.YELLOW
                        + "Warning: Auto-GPT is not running within a Docker image. This is not recommended."
                        + Fore.RESET
                    )
                if not self.is_virtual_env():
                    print(
                        Fore.YELLOW
                        + "Warning: Auto-GPT is not running within a virtualenv. This is not recommended."
                        + Fore.RESET
                    )
                print(
                    Fore.YELLOW
                    + "Warning: Unable to prompt user for container options in continuous mode. Running directly."
                    + Fore.RESET
                )

        if self._prefs == 1:
            self._run_in_docker()

        if self._prefs == 2:
            self._run_in_virtual_env()

    def prompt_for_prefs(self):
        """
        Prompts the user to select a container option.
        """
        print("Choose a container option for running Auto-GPT:")
        for option in PREFS:
            print(option)

        return int(prompt("Enter the number of your choice: "))

    def save_prefs(self) -> None:
        """
        Saves the user's preferences to user data.
        """
        logger.debug(f"Saving container prefs: {self._prefs}")
        self._prefs_file.write_text(str(self._prefs))

    def reset_prefs(self) -> None:
        """
        Resets the user's preferences.
        """
        logger.debug("Resetting container prefs.")
        self._prefs_file.unlink(missing_ok=True)
        self._prefs = None

    def _copy_to_docker(self, src: Path, dest: Path) -> bool:
        if not src.exists():
            return False
        
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy(src, dest)
        
    def _run_in_docker(self) -> None:
        """
        Runs Auto-GPT in a Docker container.
        """
        if self.is_docker():
            logger.info(
                f"Already running in Docker container. Skipping Docker initialization."
            )
            return
        logger.info("Initializing Docker container.")
        
        cwd = Path(os.getcwd())
        if not self._docker_dir.exists():
            self._docker_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing Docker container at {self._docker_dir}.")
            self._copy_to_docker(cwd / "docker-compose.yml", self._docker_dir / "docker-compose.yml")
            self._copy_to_docker(cwd / "Dockerfile", self._docker_dir / "Dockerfile")
            if not self._copy_to_docker(cwd / ".env", self._docker_dir / ".env"):
                self._copy_to_docker(cwd / ".env.template", self._docker_dir / ".env")
            if not self._copy_to_docker(cwd / "azure.yaml", self._docker_dir / "azure.yaml"):
                self._copy_to_docker(cwd / "azure.yaml.template", self._docker_dir / "azure.yaml")
        
        os.chdir(self._docker_dir)
        subprocess.run(["docker compose", "build", "autogpt"])   
        subprocess.run(
            [
                "docker",
                "run",
                "-it",
                "--rm",
                "-v",
                f"{cwd}:/app",
                "autogpt",
                "autogpt",
            ]
            + sys.argv[1:]
        )
        sys.exit(0)

    def _run_in_virtual_env(self) -> None:
        """
        Runs Auto-GPT in a virtual environment.
        """
        if self.is_virtual_env():
            logger.info(
                f"Already running in virtual environment: {sys.prefix} - Skipping virtual environment initialization."
            )
            return

        if not self._venv_dir.exists():
            self._venv_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing virtual environment at {self._venv_dir}.")
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(self._venv_dir)
        else:
            logger.info(f"Found existing virtual environment: {self._venv_dir}.")

        if sys.platform == "win32":
            python_bin = f"{self._venv_dir}/Scripts/python"
        else:
            python_bin = f"{self._venv_dir}/bin/python"

        if Path("./requirements.txt").exists():
            logger.info(f"Installing requirements from requirements.txt.")
            subprocess.run(
                [python_bin, "-m", "pip", "install", "-r", "requirements.txt"]
            )

        logger.info(f"Starting Auto-GPT in virtual environment.")
        subprocess.run([python_bin, "-m", "autogpt"] + sys.argv[1:])
        sys.exit(0)
