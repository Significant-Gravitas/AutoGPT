import logging
import os
import re
import sys

import requests
from colorama import Fore, Style
from git.repo import Repo
from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.history import InMemoryHistory

# Import custom module
from autogpt.config import Config

# Create a logger object
logger = logging.getLogger(__name)

# Create a PromptSession object for user input
session = PromptSession(history=InMemoryHistory())

# Define a function for cleaning user input
async def clean_input(config: Config, prompt: str = ""):
    try:
        if config.chat_messages_enabled:
            # Iterate through plugins to handle user input
            for plugin in config.plugins:
                if not hasattr(plugin, "can_handle_user_input"):
                    continue
                if not plugin.can_handle_user_input(user_input=prompt):
                    continue
                plugin_response = plugin.user_input(user_input=prompt)
                if not plugin_response:
                    continue
                # Process user input based on plugin responses
                if plugin_response.lower() in [
                    "yes",
                    "yeah",
                    "y",
                    "ok",
                    "okay",
                    "sure",
                    "alright",
                ]:
                    return config.authorise_key
                elif plugin_response.lower() in [
                    "no",
                    "nope",
                    "n",
                    "negative",
                ]:
                    return config.exit_key
                return plugin_response

        # Ask for user input and handle interruptions
        logger.debug("Asking user via keyboard...")
        answer = await session.prompt_async(ANSI(prompt + " "), handle_sigint=False)
        return answer
    except KeyboardInterrupt:
        logger.info("You interrupted AutoGPT")
        logger.info("Quitting...")
        exit(0)

# Define a function to fetch a bulletin from the web
def get_bulletin_from_web():
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpts/autogpt/BULLETIN.md"
        )
        if response.status_code == 200:
            return response.text
    except requests.exceptions.RequestException:
        pass

    return ""

# Define a function to get the current Git branch
def get_current_git_branch() -> str:
    try:
        repo = Repo(search_parent_directories=True)
        branch = repo.active_branch
        return branch.name
    except:
        return ""

# Define a function to get the latest bulletin
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
            "Below you'll find the latest AutoGPT News and updates regarding features!\n"
            "If you don't wish to see this message, you "
            "can run AutoGPT with the *--skip-news* flag.\n"
        )

    if new_bulletin and is_new_news:
        open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8").write(new_bulletin)
        current_bulletin = f"{Fore.RED}::NEW BULLETIN::{Fore.RESET}\n\n{new_bulletin}"

    return f"{news_header}\n{current_bulletin}", is_new_news

# Define a function to convert markdown to ANSI style
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

# Define a function to get a legal warning
def get_legal_warning() -> str:
    legal_text = """
    # DISCLAIMER AND INDEMNIFICATION AGREEMENT
    # PLEASE READ THIS DISCLAIMER AND INDEMNIFICATION AGREEMENT CAREFULLY BEFORE USING THE AUTOGPT SYSTEM. BY USING THE AUTOGPT SYSTEM, YOU AGREE TO BE BOUND BY THIS AGREEMENT.
    # Introduction
    ... (omitted for brevity) ...
    """
    return legal_text

# Define a function to print a message of the day (motd)
def print_motd(config: Config, logger: logging.Logger):
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
        if is_new_motd and not config.chat_messages_enabled:
            input(
                Fore.MAGENTA
                + Style.BRIGHT
                + "NEWS: Bulletin was updated! Press Enter to continue..."
                + Style.RESET_ALL
            )

# Define a function to print Git branch information
def print_git_branch_info(logger: logging.Logger):
    git_branch = get_current_git_branch()
    if git_branch and git_branch != "master":
        logger.warn(
            f"You are running on `{git_branch}` branch"
            " - this is not a supported branch."
        )

# Define a function to print Python version information
def print_python_version_info(logger: logging.Logger):
    if sys.version_info < (3, 10):
        logger.error(
            "WARNING: You are running on an older version of Python. "
            "Some people have observed problems with certain "
            "parts of AutoGPT with this version. "
            "Please consider upgrading to Python 3.10 or higher.",
        )
