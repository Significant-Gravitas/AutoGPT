from typing import Dict, List, Optional, Set, Union
import pika
import json
from dataclasses import dataclass
from colorama import Fore, init
import time
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.live import Live
import threading
import os

import chat
from custom_types import ChatMessage


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

@dataclass
class UserMessage:
    message: str

@dataclass
class UserAnswer:
    question: str
    answer: str


def parse_message(message: str) -> Union[UserAnswer, UserMessage]:
    """Parse a message from the user into a UserAnswer, UserMessage object."""
    message_json = json.loads(message)
    if "question" in message_json:
        if "answer" in message_json:
            return UserAnswer(message_json["question"], message_json["answer"])
        else:
            raise ValueError("Invalid message format: " + message)
    elif "message" in message_json:
        return UserMessage(message_json["message"])
    else:
        raise ValueError("Invalid message format: " + message)


class QAModel:
    """The model used by the Auto GPT Instance to ask questions and receive answers from the user."""

    def __init__(self):
        self.channel = connect_rabbitmq()
        # This class generally gets called as the first thing in the program, so lets also clear the queues
        self.channel.queue_purge(queue='togpt')
        self.channel.queue_purge(queue='touser')

    def ask_user(self, question: str) -> str:
        """Ask the user a question and return a message to the gpt agent to check back later for a response."""
        # Send the question to the user
        question_json = json.dumps({"question": question})
        self.channel.basic_publish(exchange='', routing_key='touser', body=question_json)
        return "You have asked the user a question. Please wait for a response. Do not ask the question again. You may ask other questions without waiting for a response. You may also send messages to the user without waiting for a response."

    def notify_user(self, message: str) -> str:
        """Notify the user of a message and return a message to the gpt agent to check back later for a response."""
        # Send the message to the user
        message_json = json.dumps({"message": message})
        self.channel.basic_publish(exchange='', routing_key='touser', body=message_json)
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response."

    def receive_user_response(self) -> List[ChatMessage]:
        """Checks to see if there has yet been a single response from the user and if so returns it as a JSON string."""
        out = []

        # Try to get a message from the queue
        method_frame, _, body = self.channel.basic_get(queue='togpt')

        if method_frame:
            # Acknowledge the message
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            # Return the response as a JSON string
            body_json = json.loads(body.decode())

            parsed_message = parse_message(body_json)
            if isinstance(parsed_message, UserAnswer):
                out = [
                    chat.create_chat_message("system", f"Previously you asked the user the following question: '{parsed_message.question}'. The user responded with the following answer. Please give a very high priority to this answer as it comes directly from the user."),
                    chat.create_chat_message("user", parsed_message.answer),
                ]
            elif isinstance(parsed_message, UserMessage):
                out = [
                    chat.create_chat_message("system", f"The user has sent you the following message. If it is in the form of a question, please respond with an answer in the form of a notify_user command. Please give a very high priority to this message's content as it comes directly from the user."),
                    chat.create_chat_message("user", parsed_message.message)
                ]
            else:
                raise ValueError("Invalid message type: " + str(parsed_message))

        return out

class QAClient:
    """
    The model used by the user to get questions from the Auto GPT Instance and answer them.
    This model involved thread safe operations to modify self.unanswered_questions
    """

    def __init__(self) -> None:
        self.channel = connect_rabbitmq()
        self.unanswered_questions: Set[str] = {}
        self.unanswered_questions_lock = threading.Lock()

    def receive_continuous(self) -> None:
        """Continuously meant to check for new messages from the Auto GPT Instance and add them to the list of questions that haven't been answered yet. Meant to run in a thread."""
        # Try to get a message from the queue
        while True:
            method_frame, _, body = self.channel.basic_get(queue='touser')

            if method_frame:
                # Acknowledge the message
                self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

                # Return the response as a JSON string
                body_str = body.decode()
                body_json = json.loads(body_str)
                if "question" in body_json:
                    question = body_json["question"]
                    self.add_question(question)

    def get_questions(self) -> List[str]:
        """Get a list of questions that haven't been answered yet. This is a thread-safe operation."""
        with self.unanswered_questions_lock:
            return list(sorted(list(self.unanswered_questions)))

    def add_question(self, question: str) -> None:
        """Add a question to the list of questions that haven't been answered yet. This is a thread-safe operation."""
        with self.unanswered_questions_lock:
            self.unanswered_questions.add(question)

    def remove_question(self, question: str) -> None:
        """Remove a question from the list of questions that haven't been answered yet. This is a thread-safe operation."""
        with self.unanswered_questions_lock:
            self.unanswered_questions.remove(question)

    def question_exists(self, question: str) -> bool:
        """Check if a question exists in the list of questions that haven't been answered yet. This is a thread-safe operation."""
        with self.unanswered_questions_lock:
            return question in self.unanswered_questions

    def send_response(self, message: str, question: Optional[str]) -> None:
        # Send the response to the Auto GPT Instance
        if not question:
            response = {
                "message": message
            }
        else:
            assert self.question_exists(question), "The question you are trying to answer does not exist."
            self.remove_question(question)
            response = {
                "question": question,
                "answer": message
            }
        self.channel.basic_publish(exchange='', routing_key='togpt', body=json.dumps(response))

class QAApp:
    def __init__(self):
        self.qa_client = QAClient()
        self.qa_client_thread = threading.Thread(target=self.qa_client.receive_continuous)
        self.qa_client_thread.start()
        self.console = Console()

    def _render_questions(self, questions: List[str]) -> Table:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Index")
        table.add_column("Question")

        for i, question in enumerate(questions):
            table.add_row(str(i), question)

        return table

    def run(self):
        with Live(console=self.console, refresh_per_second=1, vertical_overflow="visible") as live:
            while True:
                questions = self.qa_client.get_questions()
                live.update(self._render_questions(questions))
                time.sleep(1)

                if not self.qa_client_thread.is_alive():
                    break

            print("Press 'q' to quit, 'm' to send a message, or a number key to reply to a question.")

            while True:
                key = self.console.input("Enter command: ").lower()

                if key == "q":
                    break
                elif key == "m":
                    message = Prompt.ask("Message")
                    if message.lower() != "cancel":
                        self.qa_client.send_response(message, None)
                        print("[bold green]Message sent:[/bold green] {message}")
                    else:
                        print("[bold red]Message cancelled.[/bold red]")
                elif key.isdigit():
                    index = int(key)
                    questions = self.qa_client.get_questions()

                    if 0 <= index < len(questions):
                        question = questions[index]
                        answer = Prompt.ask(f"Answer for '{question}'")
                        self.qa_client.send_response(answer, question)
                        print(f"[bold green]Answer sent:[/bold green] {answer}")
                    else:
                        print("[bold red]Invalid question index.[/bold red]")



def main():
    app = QAApp()
    app.run()


if __name__ == "__main__":
    main()