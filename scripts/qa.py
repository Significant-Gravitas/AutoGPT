from typing import Dict, List, Optional, Set, Union
import pika
import json
from dataclasses import dataclass
from colorama import Fore, init
import time
import threading
import os
from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown
import chat
from custom_types import ChatMessage
from scripts.custom_types import RolesEnum


def heartbeat_thread(connection):
    while True:
        connection.process_data_events()
        time.sleep(5)  # send a heartbeat every 5 seconds

def connect_rabbitmq():
    """ RabbitMQ connection and channel configuration."""
    connection_parameters = pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost'), port=os.getenv('RABBITMQ_PORT', 5672), credentials=pika.PlainCredentials(os.getenv('RABBITMQ_USER', 'guest'), os.getenv('RABBITMQ_PASSWORD', 'guest')))
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    channel.queue_declare(queue='touser', arguments={'x-message-ttl': 3600000}, durable=True)
    channel.queue_declare(queue='togpt', durable=True)

    # start heartbeat thread
    thread = threading.Thread(target=heartbeat_thread, args=(connection,))
    thread.daemon = True
    thread.start()

    return channel


class QAModel:
    """The model used by the Auto GPT Instance to ask questions and receive answers from the user."""

    def __init__(self):
        self.channel = connect_rabbitmq()
        # This class generally gets called as the first thing in the program, so lets also clear the queues
        self.channel.queue_purge(queue='togpt')
        self.channel.queue_purge(queue='touser')

    def message_user(self, message: str) -> str:
        """Notify the user of a message and return a message to the gpt agent to check back later for a response."""
        # Send the message to the user
        self.channel.basic_publish(exchange='', routing_key='touser', body=message)
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response."

    def receive_user_response(self) -> List[ChatMessage]:
        """Checks to see if there has yet been a single response from the user and if so returns it as a JSON string."""
        out = []

        # Try to get a message from the queue
        method_frame, _, body = self.channel.basic_get(queue='togpt')

        if method_frame:
            # Acknowledge the message
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
            out.append(chat.create_chat_message("user", body.decode()))

        return out

class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.channel = connect_rabbitmq()
        self.message_history: List[str] = []
        self.message_history_thread_lock = threading.Lock()

    def receive_continuous(self) -> None:
        """Continuously meant to check for new messages from the Auto GPT Instance and add them to the list of questions that haven't been answered yet. Meant to run in a thread."""
        # Try to get a message from the queue
        while True:
            method_frame, _, body = self.channel.basic_get(queue='touser')

            if method_frame:
                # Acknowledge the message
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

                with self.message_history_thread_lock:
                    self.message_history.append(chat.create_chat_message("assistant", body.decode()))

    def send_message(self, message: str) -> None:
        self.channel.basic_publish(exchange='', routing_key='togpt', body=message)


def display_message_history(message_history: List[str]) -> None:
    os.system("cls" if os.name == "nt" else "clear")  # Clear console
    for message in message_history:
        print(Text.from_markup(message))

def pretty_format_message(message: ChatMessage) -> str:
    role = message["role"]
    content = message["content"]

    if role == RolesEnum.user:
        sender_color = "blue"
    elif role == RolesEnum.assistant:
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