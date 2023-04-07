from scripts.qa import connect_to_redis
from scripts import chat
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

class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.redis = connect_to_redis()
        self.message_history: List[str] = []
        self.message_history_thread_lock = threading.Lock()

    def receive_continuous(self) -> None:
        """Continuously meant to check for new messages from the Auto GPT Instance and add them to the list of questions that haven't been answered yet. Meant to run in a thread."""
        # Try to get a message from the queue
        while True:
            if self.redis.llen("touser") > 0:
                message = self.redis.rpop("touser")
                with self.message_history_thread_lock:
                    self.message_history.append(chat.create_chat_message("assistant", message))

    def send_message(self, message: str) -> None:
        self.redis.lpush("togpt", message)


def display_message_history(message_history: List[str]) -> None:
    os.system("cls" if os.name == "nt" else "clear")  # Clear console
    for message in message_history:
        print(Text.from_markup(message))

def pretty_format_message(message: chat.ChatMessage) -> str:
    role = message["role"]
    content = message["content"]

    if role == chat.RolesEnum.user:
        sender_color = "blue"
    elif role == chat.RolesEnum.assistant:
        sender_color = "red"
    else:
        sender_color = "green"

    formatted_content = str(Markdown(content))
    formatted_message = f"[{sender_color}]{role}:[/{sender_color}] {formatted_content}"
    return formatted_message

def main():
    init(autoreset=True)  # Initialize colorama

    console = Console()
    qa_client = QAClient()

    receive_thread = threading.Thread(target=qa_client.receive_continuous)
    receive_thread.daemon = True
    receive_thread.start()

    message_history_text = Text()
    with Live(Panel(message_history_text), console=console, auto_refresh=False) as live:
        while True:
            user_message = Prompt.ask("Enter your message")

            if user_message.strip() == "":  # Ignore empty messages
                continue

            with qa_client.message_history_thread_lock:
                qa_client.message_history.append(chat.create_chat_message("user", user_message))
                pretty_format_message_history = [
                    pretty_format_message(message) for message in qa_client.message_history
                ]
                message_history_text = Text("\n".join(pretty_format_message_history))
                live.update(Panel(message_history_text))

            qa_client.send_message(user_message)

if __name__ == "__main__":
    main()