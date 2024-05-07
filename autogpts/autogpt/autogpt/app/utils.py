import contextlib
import logging
import os
import re
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from colorama import Fore, Style
from git import InvalidGitRepositoryError, Repo

if TYPE_CHECKING:
    from forge.config.config import Config

logger = logging.getLogger(__name__)


def get_bulletin_from_web():
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpts/autogpt/BULLETIN.md"  # noqa: E501
        )
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException:
        pass

    return ""


def get_current_git_branch() -> str:
    try:
        repo = Repo(search_parent_directories=True)
        branch = repo.active_branch
        return branch.name
    except InvalidGitRepositoryError:
        return ""


def vcs_state_diverges_from_master() -> bool:
    """
    Returns whether a git repo is present and contains changes that are not in `master`.
    """
    paths_we_care_about = "autogpts/autogpt/autogpt/**/*.py"
    try:
        repo = Repo(search_parent_directories=True)

        # Check for uncommitted changes in the specified path
        uncommitted_changes = repo.index.diff(None, paths=paths_we_care_about)
        if uncommitted_changes:
            return True

        # Find OG AutoGPT remote
        for remote in repo.remotes:
            if remote.url.endswith(
                tuple(
                    # All permutations of old/new repo name and HTTP(S)/Git URLs
                    f"{prefix}{path}"
                    for prefix in ("://github.com/", "git@github.com:")
                    for path in (
                        f"Significant-Gravitas/{n}.git" for n in ("AutoGPT", "Auto-GPT")
                    )
                )
            ):
                og_remote = remote
                break
        else:
            # Original AutoGPT remote is not configured: assume local codebase diverges
            return True

        master_branch = og_remote.refs.master
        with contextlib.suppress(StopIteration):
            next(repo.iter_commits(f"HEAD..{master_branch}", paths=paths_we_care_about))
            # Local repo is one or more commits ahead of OG AutoGPT master branch
            return True

        # Relevant part of the codebase is on master
        return False
    except InvalidGitRepositoryError:
        # No git repo present: assume codebase is a clean download
        return False


def get_git_user_email() -> str:
    try:
        repo = Repo(search_parent_directories=True)
        return repo.config_reader().get_value("user", "email", default="")
    except InvalidGitRepositoryError:
        return ""


def get_latest_bulletin() -> tuple[str, bool]:
    exists = os.path.exists("data/CURRENT_BULLETIN.md")
    current_bulletin = ""
    if exists:
        current_bulletin = open(
            "data/CURRENT_BULLETIN.md", "r", encoding="utf-8"
        ).read()
    new_bulletin = get_bulletin_from_web()
    is_new_news = new_bulletin != "" and new_bulletin != current_bulletin

    news_header = Fore.YELLOW + "Welcome to AutoGPT!\n"
    if new_bulletin or current_bulletin:
        news_header += (
            "Below you'll find the latest AutoGPT News and feature updates!\n"
            "If you don't wish to see this message, you "
            "can run AutoGPT with the *--skip-news* flag.\n"
        )

    if new_bulletin and is_new_news:
        open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8").write(new_bulletin)
        current_bulletin = f"{Fore.RED}::NEW BULLETIN::{Fore.RESET}\n\n{new_bulletin}"

    return f"{news_header}\n{current_bulletin}", is_new_news


def markdown_to_ansi_style(markdown: str):
    ansi_lines: list[str] = []
    for line in markdown.split("\n"):
        line_style = ""

        if line.startswith("# "):
            line_style += Style.BRIGHT
        else:
            line = re.sub(
                r"(?<!\*)\*(\*?[^*]+\*?)\*(?!\*)",
                rf"{Style.BRIGHT}\1{Style.NORMAL}",
                line,
            )

        if re.match(r"^#+ ", line) is not None:
            line_style += Fore.CYAN
            line = re.sub(r"^#+ ", "", line)

        ansi_lines.append(f"{line_style}{line}{Style.RESET_ALL}")
    return "\n".join(ansi_lines)


def get_legal_warning() -> str:
    legal_text = """
## DISCLAIMER AND INDEMNIFICATION AGREEMENT
### PLEASE READ THIS DISCLAIMER AND INDEMNIFICATION AGREEMENT CAREFULLY BEFORE USING THE AUTOGPT SYSTEM. BY USING THE AUTOGPT SYSTEM, YOU AGREE TO BE BOUND BY THIS AGREEMENT.

## Introduction
AutoGPT (the "System") is a project that connects a GPT-like artificial intelligence system to the internet and allows it to automate tasks. While the System is designed to be useful and efficient, there may be instances where the System could perform actions that may cause harm or have unintended consequences.

## No Liability for Actions of the System
The developers, contributors, and maintainers of the AutoGPT project (collectively, the "Project Parties") make no warranties or representations, express or implied, about the System's performance, accuracy, reliability, or safety. By using the System, you understand and agree that the Project Parties shall not be liable for any actions taken by the System or any consequences resulting from such actions.

## User Responsibility and Respondeat Superior Liability
As a user of the System, you are responsible for supervising and monitoring the actions of the System while it is operating on your
behalf. You acknowledge that using the System could expose you to potential liability including but not limited to respondeat superior and you agree to assume all risks and liabilities associated with such potential liability.

## Indemnification
By using the System, you agree to indemnify, defend, and hold harmless the Project Parties from and against any and all claims, liabilities, damages, losses, or expenses (including reasonable attorneys' fees and costs) arising out of or in connection with your use of the System, including, without limitation, any actions taken by the System on your behalf, any failure to properly supervise or monitor the System, and any resulting harm or unintended consequences.
    """  # noqa: E501
    return legal_text


def print_motd(config: "Config", logger: logging.Logger):
    motd, is_new_motd = get_latest_bulletin()
    if motd:
        motd = markdown_to_ansi_style(motd)
        for motd_line in motd.split("\n"):
            logger.info(
                extra={
                    "title": "NEWS:",
                    "title_color": Fore.GREEN,
                    "preserve_color": True,
                },
                msg=motd_line,
            )
        if is_new_motd:
            input(
                Fore.MAGENTA
                + Style.BRIGHT
                + "NEWS: Bulletin was updated! Press Enter to continue..."
                + Style.RESET_ALL
            )


def print_git_branch_info(logger: logging.Logger):
    git_branch = get_current_git_branch()
    if git_branch and git_branch != "master":
        logger.warning(
            f"You are running on `{git_branch}` branch"
            " - this is not a supported branch."
        )


def print_python_version_info(logger: logging.Logger):
    if sys.version_info < (3, 10):
        logger.error(
            "WARNING: You are running on an older version of Python. "
            "Some people have observed problems with certain "
            "parts of AutoGPT with this version. "
            "Please consider upgrading to Python 3.10 or higher.",
        )


ENV_FILE_PATH = Path(__file__).parent.parent.parent / ".env"


def env_file_exists() -> bool:
    return ENV_FILE_PATH.is_file()


def set_env_config_value(key: str, value: str) -> None:
    """Sets the specified env variable and updates it in .env as well"""
    os.environ[key] = value

    with ENV_FILE_PATH.open("r+") as file:
        lines = file.readlines()
        file.seek(0)
        key_already_in_file = False
        for line in lines:
            if re.match(rf"^(?:# )?{key}=.*$", line):
                file.write(f"{key}={value}\n")
                key_already_in_file = True
            else:
                file.write(line)

        if not key_already_in_file:
            file.write(f"{key}={value}\n")

        file.truncate()


def is_port_free(port: int, host: str = "127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))  # Try to bind to the port
            return True  # If successful, the port is free
        except OSError:
            return False  # If failed, the port is likely in use
