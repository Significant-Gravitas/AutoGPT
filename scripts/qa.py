from typing import Dict, List, Optional, Union
import pika
import json
from dataclasses import dataclass
from colorama import Fore, init
import time
import os
import keyboard

from autogpt import chat


def connect_rabbitmq():
    """ RabbitMQ connection and channel configuration."""
    connection_parameters = pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost'), port=os.getenv('RABBITMQ_PORT', 5672), credentials=pika.PlainCredentials(os.getenv('RABBITMQ_USER', 'guest'), os.getenv('RABBITMQ_PASSWORD', 'guest')))
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    channel.queue_declare(queue='touser', arguments={'x-message-ttl': 3600000})
    channel.queue_declare(queue='togpt')
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

    def receive_user_response(self) -> List[chat.ChatMessage]:
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
    """The model used by the user to get questions from the Auto GPT Instance and answer them."""

    def __init__(self):
        self.channel = connect_rabbitmq()
        self._previous = None

    def receive(self) -> Optional[str]:
        # Block and wait until a message/question is received
        method_frame, _, body = self.channel.basic_get(queue='touser', auto_ack=True)

        while not method_frame:
            if keyboard.is_pressed('enter'):  # Check if the Enter key is pressed
                return None  # Return None to indicate that the user wants to send a message
            time.sleep(1)
            method_frame, _, body = self.channel.basic_get(queue='touser', auto_ack=True)

        body_json = json.loads(body.decode())
        if 'question' in body_json:
            self._previous = body_json['question']

        return self._previous

    def send_response(self, message: str, unprompted: bool) -> None:
        # Send the response to the Auto GPT Instance
        if unprompted:
            response = {
                "message": message
            }
        else:
            assert self._previous is not None, "You can't answer a question if you haven't received one yet."
            response = {
                "question": str(self._previous),
                "answer": message
            }
            self._previous = None

        self.channel.basic_publish(exchange='', routing_key='togpt', body=json.dumps(response))


# The main loop for the client if you run this script directly
if __name__ == "__main__":
    init(autoreset=True)

    qa_client = QAClient()

    while True:
        print(Fore.GREEN + "Waiting for question or message (Press Enter to send a message)...")
        json_received = qa_client.receive()

        if json_received is None:  # The user pressed Enter to send a message
            message = input(Fore.GREEN + "MESSAGE (type 'cancel' to begin waiting for a question again): ")
            if message.lower() != 'cancel':
                qa_client.send_response(message, unprompted=True)
            else:
                print(Fore.RED + "Message cancelled.")
        elif 'question' in json_received:
            question = json.loads(json_received)['question']
            print(Fore.YELLOW + "QUESTION: " + question)
            answer = input(Fore.GREEN + "ANSWER: ")
            qa_client.send_response(answer, unprompted=False)
        elif 'message' in json_received:
            message = json.loads(json_received)['message']
            print(Fore.YELLOW + "NOTIFICATION: " + message)