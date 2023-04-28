import os

import requests
import yaml
from colorama import Fore
from git.repo import Repo
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit import HTML
from typing import Union
from autogpt.config import Config

ANSI_BLACK = "ansiblack"
ANSI_RED = "ansired"
ANSI_GREEN = "ansigreen"
ANSI_YELLOW = "ansiyellow"
ANSI_BLUE = "ansiblue"
ANSI_MAGENTA = "ansimagenta"
ANSI_CYAN = "ansicyan"
ANSI_GRAY = "ansigray"
ANSI_BRIGHTBLACK = "ansibrightblack"
ANSI_BRIGHTRED = "ansibrightred"
ANSI_BRIGHTGREEN = "ansibrightgreen"
ANSI_BRIGHTYELLOW = "ansibrightyellow"
ANSI_BRIGHTBLUE = "ansibrightblue"
ANSI_BRIGHTMAGENTA = "ansibrightmagenta"
ANSI_BRIGHTCYAN = "ansibrightcyan"
ANSI_WHITE = "ansiwhite"

session = PromptSession(history=InMemoryHistory())

def send_chat_message_to_user(report: str):
    cfg = Config()
    if not cfg.chat_messages_enabled:
        return
    for plugin in cfg.plugins:
        if not hasattr(plugin, "can_handle_report"):
            continue
        if not plugin.can_handle_report():
            continue
        plugin.report(report)

def clean_input(prompt: Union[str, FormattedText] = "", color: str = None, talk=False):
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
        print("Asking user via keyboard...")
        if color:
            prompt = HTML(f"<{color}>{prompt}</{color}>")
        user_input = session.prompt(prompt)
        return user_input.strip()
    except KeyboardInterrupt:
        print("You interrupted Auto-GPT")
        print("Quitting...")
        exit(0)
    except EOFError:
        return ""


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


def get_latest_bulletin() -> str:
    exists = os.path.exists("CURRENT_BULLETIN.md")
    current_bulletin = ""
    if exists:
        current_bulletin = open("CURRENT_BULLETIN.md", "r", encoding="utf-8").read()
    new_bulletin = get_bulletin_from_web()
    is_new_news = new_bulletin != current_bulletin

    if new_bulletin and is_new_news:
        open("CURRENT_BULLETIN.md", "w", encoding="utf-8").write(new_bulletin)
        return f" {Fore.RED}::UPDATED:: {Fore.CYAN}{new_bulletin}{Fore.RESET}"
    return current_bulletin
