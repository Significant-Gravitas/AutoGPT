import os
import re

import requests
import yaml
from colorama import Fore, Style
from git.repo import Repo

from autogpt.logs import logger
from autogpt.commands.command import CommandRegistry

# Use readline if available (for clean_input)
try:
    import readline
except ImportError:
    pass

from autogpt.config import Config

import click

from prompt_toolkit import prompt as prompt_tk
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

def shell_input(query, agent):
    # Define a list of valid commands
    # TODO receive from command mgr

    valid_commands = [cmd.name for cmd in agent.command_registry.get_commands()]


    # Define a WordCompleter with the valid commands
    command_completer = WordCompleter(valid_commands)

    # Set up the prompt with history and auto-completion
    history_file = ".my_shell_history"
    history = FileHistory(history_file)
    while True:
        # query = click.style(query, fg="magenta")
        return prompt_tk(query, history=history, auto_suggest=AutoSuggestFromHistory(), completer=command_completer)
        if line in valid_commands:
            print(f"Executing {line}")
            if line == "exit":
                exit()
        else:
            print(f"Invalid command: {line}")



def clean_input(prompt: str, agent):
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
        answer = shell_input(prompt, agent) # input(prompt)
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
