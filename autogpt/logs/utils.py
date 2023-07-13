from __future__ import annotations

from typing import TYPE_CHECKING

from colorama import Fore

if TYPE_CHECKING:
    from autogpt.config import Config

from .logger import logger


def print_assistant_thoughts(
    ai_name: str,
    assistant_reply_json_valid: dict,
    config: Config,
) -> None:
    from autogpt.speech import say_text

    assistant_thoughts_reasoning = None
    assistant_thoughts_plan = None
    assistant_thoughts_speak = None
    assistant_thoughts_criticism = None

    assistant_thoughts = assistant_reply_json_valid.get("thoughts", {})
    assistant_thoughts_text = remove_ansi_escape(assistant_thoughts.get("text", ""))
    if assistant_thoughts:
        assistant_thoughts_reasoning = remove_ansi_escape(
            assistant_thoughts.get("reasoning", "")
        )
        assistant_thoughts_plan = remove_ansi_escape(assistant_thoughts.get("plan", ""))
        assistant_thoughts_criticism = remove_ansi_escape(
            assistant_thoughts.get("criticism", "")
        )
        assistant_thoughts_speak = remove_ansi_escape(
            assistant_thoughts.get("speak", "")
        )
    logger.typewriter_log(
        f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text
    )
    logger.typewriter_log("REASONING:", Fore.YELLOW, str(assistant_thoughts_reasoning))
    if assistant_thoughts_plan:
        logger.typewriter_log("PLAN:", Fore.YELLOW, "")
        # If it's a list, join it into a string
        if isinstance(assistant_thoughts_plan, list):
            assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
        elif isinstance(assistant_thoughts_plan, dict):
            assistant_thoughts_plan = str(assistant_thoughts_plan)

        # Split the input_string using the newline character and dashes
        lines = assistant_thoughts_plan.split("\n")
        for line in lines:
            line = line.lstrip("- ")
            logger.typewriter_log("- ", Fore.GREEN, line.strip())
    logger.typewriter_log("CRITICISM:", Fore.YELLOW, f"{assistant_thoughts_criticism}")
    # Speak the assistant's thoughts
    if assistant_thoughts_speak:
        if config.speak_mode:
            say_text(assistant_thoughts_speak, config)
        else:
            logger.typewriter_log("SPEAK:", Fore.YELLOW, f"{assistant_thoughts_speak}")


def remove_ansi_escape(s: str) -> str:
    return s.replace("\x1B", "")
