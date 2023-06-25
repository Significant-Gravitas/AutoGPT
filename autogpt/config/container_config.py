"""
ContainerConfig class to enable the user to choose between Docker and virtual environment, and start Auto-GPT in the chosen environment.
"""
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv
import click
from platformdirs import user_config_dir
from colorama import Fore
from prompt_toolkit import prompt
import requests
from autogpt.logs import logger

PREFS_FILE = ".container_prefs"

PREFS = [
    "1. Always use Docker: Recommended, most secure. (Requires Docker to be installed.)",
    "2. Always use virtual environment: Avoids dependency conflicts.",
    "3. Run Auto-GPT directly and ask again next time.",
    "4. Run Auto-GPT directly and never ask again. Use --container-options to change this.",
]

FILES_LIST = [
    "docker-compose.yml.template",
    "plugins_config.yaml",
    "prompt_settings.yaml",
    ".env.template",
    "azure.yaml.template",
    "requirements.txt",
]

FILES_REPO = "Significant-Grativas/Auto-GPT"
FILES_BRANCH = "stable"

class ContainerConfig:
    def __init__(
        self,
        image_name: str,
        repo: str, 
        branch_or_tag: str,
        rebuild_image: bool, 
        pull_image: bool,
        reinstall: bool, 
        interactive: bool, 
        allow_virtualenv: bool,
        args=list[str]
    ):
        """
        Initlaizes the ContainerConfig class.
        """
        self.image_name = image_name
        self.repo = repo
        self.branch_or_tag = branch_or_tag
        self.rebuild_image = rebuild_image
        self.pull_image = pull_image
        self.reinstall = reinstall
        self.interactive = interactive
        self.allow_virtualenv = allow_virtualenv
        self.args = args
        
        self.config_dir = Path(user_config_dir("autogpt"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.prefs_file = Path(self.config_dir, PREFS_FILE)
        self.venv_dir = self.config_dir / "venv"
        self.docker_config_dir = self.config_dir / "docker"
            
        try:
            self.prefs = (
                int(self.prefs_file.read_text()) if self.prefs_file.exists() else None
            )
        except ValueError:
            self.prefs = None

        logger.debug(
            f"ContainerConfig initialized with prefs: {self.prefs} [{PREFS[self.prefs-1] if self.prefs else None}]"
        )

    def is_running_in_docker(self):
        """
        Returns True if Auto-GPT is running in a Docker container.
        """
        return Path("/.dockerenv").exists()

    def is_running_in_virtual_env(self):
        """
        Returns True if Auto-GPT is running in a virtual environment.
        """
        return hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        
    def run(self) -> None:
        """
        Runs the container configuration process if not running in a Docker container.
        If virtual environment is allowed, runs the selection process. Otherwise, runs the docker configuration process
        for Docker.
        """
        if self.is_running_in_docker():
            logger.info("Running in Docker container. Skipping container config.")
            return

        if self.allow_virtualenv:
            self.show_selections()
        else:
            # Docker is the only option
            assert self.check_docker_is_installed(), "Docker is not installed. Please install Docker and try again."
                
            if self.reinstall:
                self.run_reinstall()
            elif self.rebuild_image:
                self.run_rebuild_image()
            elif self.pull_image:
                self.run_pull_image()
            else:
                self.run_in_docker()
    
    def _install_docker_linux(self)-> bool:
        """
        Installs Docker on Linux using the apt package manager.

        Returns:
            bool: True if Docker was installed successfully, False otherwise.
        """
        logger.info("Installing Docker on Linux...")
        try:
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(
                [
                    "sudo",
                    "apt-get",
                    "install",
                    "-y",
                    "apt-transport-https",
                    "ca-certificates",
                    "curl",
                    "gnupg",
                    "lsb-release",
                ]
            )

            subprocess.check_call(
                [
                    "curl",
                    "-fsSL",
                    "https://download.docker.com/linux/ubuntu/gpg",
                    "|",
                    "sudo",
                    "gpg",
                    "--dearmor",
                    "-o",
                    "/usr/share/keyrings/docker-archive-keyring.gpg",
                ]
            )

            subprocess.check_call(
                [
                    "echo",
                    "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable",
                    "|",
                    "sudo",
                    "tee",
                    "/etc/apt/sources.list.d/docker.list",
                    "> /dev/null",
                ]
            )

            # Update the apt package list (for the new apt repo)
            subprocess.check_call(["sudo", "apt-get", "update"])
            subprocess.check_call(
                [
                    "sudo",
                    "apt-get",
                    "install",
                    "-y",
                    "docker-ce",
                    "docker-ce-cli",
                    "containerd.io",
                ]
            )
            logger.info("Docker has been installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.info("Docker installation failed: ", e)
            return False
            
    def _install_docker_mac(self) -> bool:
        """
        Installs Docker on MacOS using the Homebrew package manager.

        Returns:
            bool: True if Docker was installed successfully, False otherwise.
        """
        logger.info("Installing Docker on MacOS...")
        try:
            if not shutil.which("brew"):
                logger.info("Homebrew is not installed. Manual install required.")
                return False
            subprocess.check_call(["brew", "install", "docker"])
            logger.info("Docker has been installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.info("Docker installation failed: ", e)
            return False


    def _install_docker_windows(self)-> bool:
        """
        Installs Docker on Windows using the Chocolatey package manager.

        Returns:
            bool: True if Docker was installed successfully, False otherwise.
        """
        logger.info("Installing Docker on Windows...")
        try:
            if not shutil.which("choco"):
                logger.info("Chocolatey is not installed. Manual install required.")
                return False
            subprocess.check_call(["choco", "install", "docker-desktop"])
            logger.info("Docker has been installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.info("Docker installation failed: ", e)
            return False

    def install_docker(self) -> bool:
        """
        Installs Docker on the specified operating system.
        
        Returns:
            bool: True if Docker was installed successfully, False otherwise.
        """
        logger.info("Installing Docker... You may be required to enter an admin password.")
        assert not self.docker_is_installed(), "Docker is already installed."
        
        if sys.platform.startswith("linux"):
            return self._install_docker_linux()
        elif sys.platform.startswith("darwin"):
            return self._install_docker_mac()
        elif sys.platform.startswith("win32"):
            return self._install_docker_windows()
        return False
    
    def check_docker_is_installed(self, install=True) -> bool:
        """
        Checks if Docker is installed on the system. If not, prompts the user to install Docker and installs it if the user agrees.
        If Docker is already installed, this function does nothing.

        Args:
            install (boolean): Whether to install Docker if it is not already installed. Defaults to True.
         """

        if self.docker_is_installed():
            return True
        else:
            print(
                """
                Docker is not installed on your system. Docker is required to use this script.
                For non-Docker options, please refer to the documentation at https://docs.agpt.co/.
                If you have Docker Desktop installed, please make sure it is running and then try this script again.
                You can set Docker Desktop to run on startup in its settings.
                """
            )
        
        if install:
            if self.interactive and not click.confirm("Do you want to try and install Docker now?", default=True):
                logger.warn(
                    "Docker is not installed. Run this script again when you have Docker installed. Exiting..."
                )
                return False

            if self.install_docker():
                logger.info("Docker has been installed successfully.")
                return True
            else:
                logger.warn(
                    "Docker installation failed. Please install Docker and try again."
                    "You can download Docker from https://docs.docker.com/get-docker/."
                )
                return False
        else:
            return False
            
    def docker_is_installed(self) -> bool:
        """
        Checks if Docker is installed on the system.

        Returns:
            bool: True if Docker is installed, False otherwise.
        """
        try:
            subprocess.run(["docker", "--version"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
        
    def show_selections(self) -> None:
        """
        Checks if the user has not selected any container preferences and prompts the user to select one if so.
        Otherwise, it initializes the container based on the user's preferences.
        """
        
        assert not self.is_running_in_docker(), "Running in Docker container. Skipping selections."
        assert self.allow_virtualenv, "Virtualenv is not allowed. Skipping selections."
        
        if self.prefs is None or self.prefs == 3:
            print(
                Fore.YELLOW
                + "YOU HAVE NOT SELECTED A CONTAINER OPTION YET.\n"
                + "- It recommended to run Auto-GPT within either a docker container or a virtualenv for increased security and to avoid dependency conflicts.\n"
                + Fore.RESET
            )
            if self.interactive:
                print(
                    Fore.YELLOW
                    + "This prompt will not appear again, but you can use --container-options to change your selection later.\n"
                    + Fore.RESET
                )
                self.prefs = self.prompt_for_prefs()
                self.save_prefs()
            else:
                print(
                    Fore.YELLOW
                    + "You can use --container-options to make a selection at any time.\n"
                    + Fore.RESET
                )
                if not self.is_running_in_docker():
                    print(
                        Fore.YELLOW
                        + "Warning: Auto-GPT is not running within a Docker image. This is not recommended."
                        + Fore.RESET
                    )
                if not self.is_running_in_virtual_env():
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

        if self.prefs == 1:
            return self.run_in_docker()

        if self.prefs == 2:
            return self.run_in_virtual_env()

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
        logger.debug(f"Saving container prefs: {self.prefs}")
        self.prefs_file.write_text(str(self.prefs))

    def reset_prefs(self) -> None:
        """
        Resets the user's preferences.
        """
        logger.debug("Resetting container prefs.")
        self.prefs_file.unlink(missing_ok=True)
        self.prefs = None

    def copy_file(self, src: Path, dest: Path) -> bool:
        if not src.exists():
            return False
        
        if src.is_dir():
            dest_path = shutil.copytree(src, dest)
            return dest_path.exists() and dest_path.is_dir() and dest_path == dest
        else:
            dest_path = shutil.copy(src, dest)
            return dest_path.exists() and dest_path.is_file() and dest_path == dest
            
    
    def copy_config_files(self, cwd: Path, files_list: list[str] = FILES_LIST) -> None:
        uncopied_files = []
        for file_name in files_list:
            dest_file = self.docker_config_dir / file_name
            if dest_file.exists():
                logger.info(f"File {file_name} already exists. Skipping to avoid overwriting.")
                continue
            
            if not self.copy_file(cwd / file_name, self.docker_config_dir / file_name):
                uncopied_files.append(file_name)
        
            if file_name.endswith(".template"):
                self.copy_template_file(file_name)
        if uncopied_files:
            self.download_config_files(self.base_url, uncopied_files)
            
    def copy_template_file(self, file_name: str) -> str:
        if not file_name.endswith(".template"):
            logger.info(f"File {file_name} is not a template. Skipping copying.")
        dest_file = self.docker_config_dir / file_name.replace(".template", "")
        if dest_file.exists():
            logger.info(f"File {file_name} already exists. Skipping copying template to avoid overwriting.")
            
        final_path = shutil.copy(self.docker_config_dir / file_name, dest_file)
        return final_path.exists() and final_path.is_file() and final_path == dest_file

    def get_files_base_url(self) -> str:
        """
        Prompts the user for the GitHub user, repo, and branch/tag to construct the base URL for downloading files.

        Returns:
            str: The base URL for downloading files.
        """
        github_files_base = "https://raw.githubusercontent.com"
        
        if not self.repo:
            self.repo = click.prompt("GitHub 'repo-user/repo-name'", type=str, default=FILES_REPO) if self.interactive else FILES_REPO
        
        if not self.branch_or_tag:
            self.branch_or_tag = click.prompt("GitHub branch or tag", type=str, default=FILES_BRANCH) if self.interactive else FILES_BRANCH
        
        return f"{github_files_base}/{self.repo}/{self.branch_or_tag}/"
        
    def download_config_files(self, files_list: list[str] = FILES_LIST) -> None:
        """
        Downloads files from a given base URL and saves them to the user's configuration directory.
        If a downloaded file is a template, it is copied to a non-template version.

        Args:
            files_list (list[str]): A list of files to download. Defaults to FILES_LIST.
        """
        base_url = self.get_files_base_url()
        for file_name in files_list:
            url = f"{base_url}/{file_name}"
            try:
                local_file = Path(self.docker_config_dir, file_name)
                if local_file.exists():
                    logger.info(f"File {file_name} already exists. Skipping download to avoid overwriting.")
                    continue
                
                response = requests.get(url)
                response.raise_for_status()
                local_file.write_text(response.text)

                if re.search(r"4[0-9][0-9]: \[*\]", response.text):
                    raise RuntimeError(
                        f"Error downloading {url}. Please check the URL and try again.\n{response.text}"
                    )

                if file_name.endswith(".template"):
                    self.copy_template_file(file_name)
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Error downloading {url}. Please check the URL and try again.\n{e}"
                )
                
    def run_reinstall(self) -> None:
        """
        Reinstalls the container configuration.
        """
        logger.info("Reinstalling container configuration...")
        logger.info("Backing up config files...")
        backup_dir = Path(self.config_dir + f"-backup-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(self.config_dir, backup_dir)
        self.config_dir.unlink(missing_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.run_in_docker()
        
    def run_pull_image(self) -> None:
        """
        Pulls the Auto-GPT image from Docker Hub.
        """
        logger.info("Pulling Auto-GPT image from Docker Hub...")
        assert self.check_docker_is_installed(), "Docker is not installed. Please install Docker and try again."
        self.init_docker_config()
        os.chdir(self.docker_config_dir)
        subprocess.run(["docker", "pull", self.image_name], cwd=self.docker_config_dir, check=True)
        logger.info("Image pull complete.")
                
    def run_rebuild_image(self) -> None:
        logger.info("Rebuilding Auto-GPT image...")
        assert self.check_docker_is_installed(), "Docker is not installed. Please install Docker and try again."
        self.init_docker_config()
        os.chdir(self.docker_config_dir)
        subprocess.run(
            ["docker", "build", "-t", self.image_name, "."], cwd=self.docker_config_dir, check=True
        )
        logger.info("Image rebuild complete.")
        
    def init_docker_config(self) -> None:
        cwd = Path(os.getcwd())
        if not self.docker_config_dir.exists():
            self.docker_config_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initializing Docker container at {self.docker_config_dir}.")
        self.copy_config_files(cwd)
        
    def run_in_docker(self) -> None:
        """
        Runs Auto-GPT in a Docker container.
        """
        assert not self.is_running_in_docker(), "Already running in Docker container. Skipping Docker initialization."
        assert self.check_docker_is_installed(), "Docker is not installed. Please install Docker and try again."
        
        logger.info("Initializing Docker container.")
        
        self.init_docker_config()
        os.chdir(self.docker_config_dir)
        subprocess.run(
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "auto-gpt",
            ]
            + self.args,
        )
        sys.exit(0)

    def run_in_virtual_env(self) -> None:
        """
        Runs Auto-GPT in a virtual environment.
        """
        assert not self.is_running_in_virtual_env(), f"Already running in virtual environment: {sys.prefix} - Skipping virtual environment initialization."

        if not self.venv_dir.exists():
            self.venv_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing virtual environment at {self.venv_dir}.")
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(self.venv_dir)
        else:
            logger.info(f"Found existing virtual environment: {self.venv_dir}.")

        if sys.platform == "win32":
            python_bin = f"{self.venv_dir}/Scripts/python"
        else:
            python_bin = f"{self.venv_dir}/bin/python"

        if Path("./requirements.txt").exists():
            logger.info(f"Installing requirements from requirements.txt.")
            subprocess.run(
                [python_bin, "-m", "pip", "install", "-r", "requirements.txt"]
            )

        logger.info(f"Starting Auto-GPT in virtual environment.")
        subprocess.run([python_bin, "-m", "autogpt"] + sys.argv[1:])
        sys.exit(0)
