import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
import requests
from platformdirs import user_config_dir

CONFIG_DIR = Path(user_config_dir("autogpt"))

FILES_TO_DOWNLOAD = [
    "docker-compose.yml.template",
    "plugins_config.yaml",
    "prompt_settings.yaml",
    ".env.template",
    "azure.yaml.template",
    "requirements.txt",
]

DOCKER_IMAGE = "significantgravitas/auto-gpt"


def ensure_valid_os() -> str:
    """
    Returns the operating system name as a string.

    Returns:
        str: The operating system name, which can be one of "LINUX", "MAC", or "WINDOWS".

    Raises:
        SystemExit: If the operating system is not Linux, MacOS, or Windows.
    """
    if sys.platform.startswith("linux"):
        return "LINUX"
    elif sys.platform.startswith("darwin"):
        return "MAC"
    elif sys.platform.startswith("win32"):
        return "WINDOWS"
    else:
        raise SystemExit(
            "This script is only compatible with Linux, MacOS and Windows."
        )


def docker_is_installed() -> bool:
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


def ensure_docker_cmd(os_type: str) -> None:
    """
    Checks if Docker is installed on the system. If not, prompts the user to install Docker and installs it if the user agrees.
    If Docker is already installed, this function does nothing.

    Args:
        os_type (str): The operating system name, which can be one of "LINUX", "MAC", or "WINDOWS".

    Raises:
        SystemExit: If Docker installation fails or the user chooses not to install Docker.
    """

    if not docker_is_installed():
        print(
            """
            Docker is not installed on your system. Docker is required to use this script.
            For non-Docker options, please refer to the documentation at https://docs.agpt.co/.
            If you have Docker Desktop installed, please make sure it is running and then try this script again.
            You can set Docker Desktop to run on startup in its settings.
            """
        )

        if not click.confirm(
            "Do you want to try and install Docker now?", default=True
        ):
            raise SystemExit(
                "Docker is not installed. Run this script again when you have Docker installed. Exiting..."
            )

        if not install_docker(os_type):
            raise SystemExit(
                "Docker installation failed. Please install Docker and try again."
                "You can download Docker from https://docs.docker.com/get-docker/."
            )


def download_files(base_url: str) -> None:
    """
    Downloads files from a given base URL and saves them to the user's configuration directory.
    If a downloaded file is a template, it is copied to a non-template version.

    Args:
        base_url (str): The base URL to download files from.

    Raises:
        SystemExit: If there is an error downloading a file.
    """
    for file in FILES_TO_DOWNLOAD:
        url = f"{base_url}/{file}".replace("//", "/")
        try:
            response = requests.get(url)
            response.raise_for_status()
            local_file = Path(CONFIG_DIR, file)
            local_file.write_text(response.text)

            if re.search(r"4[0-9][0-9]: \[*\]", response.text):
                raise SystemExit(
                    f"Error downloading {url}. Please check the URL and try again.\n{response.text}"
                )

            if file.endswith(".template"):
                local_file = Path(CONFIG_DIR, file.replace(".template", ""))
                shutil.copyfile(Path(CONFIG_DIR, file), local_file)
        except requests.exceptions.RequestException as e:
            raise SystemExit(
                f"Error downloading {url}. Please check the URL and try again.\n{e}"
            )


def save_openai_api_key() -> bool:
    """
    Prompts the user to enter their OpenAI API key and saves it to the .env file in the user's configuration directory.

    Returns:
        bool: True if the API key was successfully saved, False otherwise.
    """
    api_key = click.prompt("Open API Key: ")
    env_path = CONFIG_DIR / ".env"
    env_path.write_text(env_path.read_text().replace("your-openai-api-key", api_key))
    return True


def pull_docker_image() -> None:
    """
    Pulls the Docker image specified by the DOCKER_IMAGE constant.

    Raises:
        CalledProcessError: If there is an error pulling the Docker image.
    """
    image_name = DOCKER_IMAGE
    try:
        subprocess.check_call(["docker", "pull", image_name])
        click.echo(f"Successfully pulled {image_name}.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to pull {image_name}.\n{e}")


def install_launch_script(base_url: str) -> None:
    """
    Downloads the autogpt command script from the specified base URL and installs it in the current directory.

    Args:
        base_url (str): The base URL to download the autogpt command script from.

    Raises:
        SystemExit: If the autogpt command script fails to install.
    """
    try:
        launcher_cmd_src = f"{base_url}/scripts/autogpt_cmd.sh".replace("//", "/")
        response = requests.get(launcher_cmd_src)
        response.raise_for_status()
        autogpt_cmd_file = Path("autogpt")
        autogpt_cmd_file.write_text(response.text)
        os.chmod(autogpt_cmd_file, 0o755)
        subprocess.run(["sudo", "cp", "autogpt", "/usr/local/bin/"], check=True)
        click.echo("Successfully installed autogpt command.")
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Error downloading {launcher_cmd_src}.\n{e}")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Error installing autogpt command.\n{e}")


def launch_auto_gpt(config_dir: str) -> None:
    """
    Launches the AutoGPT command from the specified configuration directory.

    Args:
        config_dir (str): The path to the configuration directory.

    Raises:
        SystemExit: If there is an error launching AutoGPT.
    """
    try:
        os.chdir(config_dir)
        subprocess.check_call([f"./autogpt"])
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Error launching AutoGPT.\n{e}")


def install_docker_linux()-> bool:
    """
    Installs Docker on Linux using the apt package manager.

    Returns:
        bool: True if Docker was installed successfully, False otherwise.
    """
    click.echo("Installing Docker on Linux...")
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
        click.echo("Docker has been installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        click.echo("Docker installation failed: ", e)
        return False


def install_docker_mac() -> bool:
    """
    Installs Docker on MacOS using the Homebrew package manager.

    Returns:
        bool: True if Docker was installed successfully, False otherwise.
    """
    click.echo("Installing Docker on MacOS...")
    try:
        if not shutil.which("brew"):
            click.echo("Homebrew is not installed. Manual install required.")
            return False
        subprocess.check_call(["brew", "install", "docker"])
        click.echo("Docker has been installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        click.echo("Docker installation failed: ", e)
        return False


def install_docker_windows()-> bool:
    """
    Installs Docker on Windows using the Chocolatey package manager.

    Returns:
        bool: True if Docker was installed successfully, False otherwise.
    """
    click.echo("Installing Docker on Windows...")
    try:
        if not shutil.which("choco"):
            click.echo("Chocolatey is not installed. Manual install required.")
            return False
        subprocess.check_call(["choco", "install", "docker-desktop"])
        click.echo("Docker has been installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        click.echo("Docker installation failed: ", e)
        return False


def install_docker(os_type: str) -> bool:
    """
    Installs Docker on the specified operating system.

    Args:
        os_type (str): The operating system type. Must be one of "LINUX", "MAC", or "WINDOWS".

    Returns:
        bool: True if Docker was installed successfully, False otherwise.
    """
    click.echo("Installing Docker... You may be required to enter an admin password.")
    if os_type == "LINUX":
        return install_docker_linux()
    elif os_type == "MAC":
        return install_docker_mac()
    elif os_type == "WINDOWS":
        return install_docker_windows()
    return False


def get_files_base_url() -> str:
    """
    Prompts the user for the GitHub user, repo, and branch/tag to construct the base URL for downloading files.

    Returns:
        str: The base URL for downloading files.
    """
    github_files_base = "https://raw.githubusercontent.com"
    github_user = click.prompt("GitHub user", type=str, default="Significant-Gravitas")
    github_repo = click.prompt("GitHub repo", type=str, default="Auto-GPT")
    github_branch_or_tag = click.prompt(
        "GitHub branch or tag", type=str, default="stable"
    )
    return f"{github_files_base}/{github_user}/{github_repo}/{github_branch_or_tag}/".replace(
        "//", "/"
    )


def main() -> None:
    """
    The main function of the Auto-GPT installation script. 
    This function ensures that the script is not being run inside a Docker container, 
    installs Docker on the user's system, downloads necessary files, 
    saves the OpenAI API key, pulls the Docker image, installs the launch script, and launches Auto-GPT.

    Returns:
        None
    """
    try:
        if os.path.exists("/.dockerenv"):
            raise SystemExit("This script should not be run inside a Docker container.")

        os_type = ensure_valid_os()
        ensure_docker_cmd(os_type)
        config_dir = Path(CONFIG_DIR)
        config_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(config_dir)
        base_files_url = get_files_base_url()

        download_files(base_files_url)
        save_openai_api_key()
        pull_docker_image()

        install_launch_script(base_files_url)
        launch_auto_gpt()
    except KeyboardInterrupt:
        click.echo("\n\nExiting...")
        sys.exit(1)
    except SystemExit as e:
        click.echo(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
