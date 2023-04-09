import json
from colorama import init
from scripts.qa import connect_to_redis
from scripts import chat
from scripts import speak
from typing import List
import threading
from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.console import Console
import os
from uuid import uuid4
import datetime
import argparse
from pathlib import Path

# Create an argparse that has the boolean parameter --speak
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
args = parser.parse_args()

# Get the current timestamp
now = datetime.datetime.now()

# Format the timestamp as a string
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")

# Create a Path object with the timestamp
path = Path("outputs/logs")
assert path.is_dir(), "The logs directory doesn't exist."
LOG_PATH = path/f"{timestamp}.jsonl"
assert not path.is_file(), "Somehow the log file already exists."

class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.redis = connect_to_redis()

    def receive_continuous(self, console: Console, console_lock: threading.Lock) -> None:
        """Continuously meant to check for new messages from the Auto GPT Instance and add them to the list of questions that haven't been answered yet. Meant to run in a thread."""
        # Try to get a message from the queue
        while True:
            if self.redis.llen("touser") > 0:
                message = self.redis.rpop("touser")
                with console_lock:
                    msg = chat.create_chat_message("assistant", message)
                    with open(LOG_PATH, "a") as f:
                        f.write(json.dumps(msg)+"\n")
                    if args.speak:
                        speak.say_text(msg["content"])
                    console.print(pretty_format_message(msg))

    def send_message(self, message: str) -> None:
        self.redis.lpush("togpt", message)


def pretty_format_message(message: chat.ChatMessage) -> Markdown:
    role = message["role"]
    content = message["content"]

    formatted_message = f"**[{role.name.capitalize()}]**: {content}"
    return Markdown(formatted_message)

def main():
    init(autoreset=True)  # Initialize colorama

    qa_client = QAClient()

    console = Console()
    console_lock = threading.Lock()

    receive_thread = threading.Thread(target=qa_client.receive_continuous, args=(console, console_lock))
    receive_thread.daemon = True
    receive_thread.start()

    console.print("Welcome to the Auto GPT Client!")
    console.print("Type your messages below and press enter to send them to the Auto GPT Instance.")
    console.print("You can also type \"exit\" to exit the client.")
    console.print("")

    while True:
        message = console.input("")
        if message:
            if message == "exit":
                break
            qa_client.send_message(message)
            with console_lock:
                msg = chat.create_chat_message("user", message)
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps(msg))

    console.print("Exiting client...")

if __name__ == "__main__":
    main()