import os
import re

import requests
import yaml
from colorama import Fore, Style
from git.repo import Repo

from autogpt.logs import logger

# Use readline if available (for clean_input)
try:
    import readline
except ImportError:
    pass

from autogpt.config import Config


def batch(iterable, max_batch_length: int, overlap: int = 0):
    """Batch data from iterable into slices of length N. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if max_batch_length < 1:
        raise ValueError("n must be at least one")
    for i in range(0, len(iterable), max_batch_length - overlap):
        yield iterable[i : i + max_batch_length]


def clean_input(prompt: str = "", talk=False):
    try:
        cfg = Config()
        if cfg.chat_messages_enabled:
            for plugin in cfg.plugins:
                if not hasattr(plugin, "can_handle_user_input"):
                    continue
                if not plugin.can_handle_user_input(user_input=prompt):
                    continue
                plugin_response = plugin.user_input(user_input=prompt)
                if not plugin_response:
                    continue
                if plugin_response.lower() in [
                    "yes",
                    "yeah",
                    "y",
                    "ok",
                    "okay",
                    "sure",
                    "alright",
                ]:
                    return cfg.authorise_key
                elif plugin_response.lower() in [
                    "no",
                    "nope",
                    "n",
                    "negative",
                ]:
                    return cfg.exit_key
                return plugin_response

        # ask for input, default when just pressing Enter is y
        logger.info("Asking user via keyboard...")
        answer = input(prompt)
        return answer
    except KeyboardInterrupt:
        logger.info("You interrupted Auto-GPT")
        logger.info("Quitting...")
        exit(0)


def validate_yaml_file(file: str):
    try:
        with open(file, encoding="utf-8") as fp:
            yaml.load(fp.read(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        return (False, f"The file {Fore.CYAN}`{file}`{Fore.RESET} wasn't found")
    except yaml.YAMLError as e:
        return (
            False,
            f"There was an issue while trying to read with your AI Settings file: {e}",
        )

    return (True, f"Successfully validated {Fore.CYAN}`{file}`{Fore.RESET}!")


def readable_file_size(size, decimal_places=2):
    """Converts the given size in bytes to a readable format.
    Args:
        size: Size in bytes
        decimal_places (int): Number of decimal places to display
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def get_bulletin_from_web():
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/master/BULLETIN.md"
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
    except:
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

    news_header = Fore.YELLOW + "Welcome to Auto-GPT!\n"
    if new_bulletin or current_bulletin:
        news_header += (
            "Below you'll find the latest Auto-GPT News and updates regarding features!\n"
            "If you don't wish to see this message, you "
            "can run Auto-GPT with the *--skip-news* flag.\n"
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
            """
    return legal_text
