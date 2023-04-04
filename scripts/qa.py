from typing import Dict, Optional
import pika
import json
from dataclasses import dataclass
from colorama import Fore, init
import time
import os
import keyboard


def connect_rabbitmq():
    """ RabbitMQ connection and channel configuration."""
    connection_parameters = pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'localhost'), port=os.getenv('RABBITMQ_PORT', 5672), credentials=pika.PlainCredentials(os.getenv('RABBITMQ_USER', 'guest'), os.getenv('RABBITMQ_PASSWORD', 'guest')))
    connection = pika.BlockingConnection(connection_parameters)
    channel = connection.channel()
    channel.queue_declare(queue='touser', arguments={'x-message-ttl': 3600000})
    channel.queue_declare(queue='togpt')
    return channel


class QAModel:
    """The model used by the Auto GPT Instance to ask questions and receive answers from the user."""

    def __init__(self):
        self.channel = connect_rabbitmq()

    def ask_user(self, question: str) -> None:
        """Ask the user a question and return a message to the gpt agent to check back later for a response."""
        # Send the question to the user
        question_json = json.dumps({"question": question})
        self.channel.basic_publish(exchange='', routing_key='touser', body=question_json)
        return None

    def notify_user(self, message: str) -> None:
        """Notify the user of a message and return a message to the gpt agent to check back later for a response."""
        # Send the message to the user
        message_json = json.dumps({"message": message})
        self.channel.basic_publish(exchange='', routing_key='touser', body=message_json)
        return None

    def receive_user_response(self) -> List[str]:
        """Checks to see if there has yet been a single response from the user and if so returns it as a JSON string."""
        out = []

        # Try to get a message from the queue
        method_frame, _, body = self.channel.basic_get(queue='togpt')

        if method_frame:
            # Acknowledge the message
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            # Return the response as a JSON string
            body_json = json.loads(body.decode())

            if 'question' in body_json:
                out.append("QUESTION: "+body_json['question'])
            if 'answer' in body_json:
                out.append("ANSWER: "+body_json['answer'])
            if 'message' in body_json:
                out.append("MESSAGE: "+body_json['message'])

            # Check that the message is valid
            # Check that if there is an answer, there is also a question
            assert all(key in body_json for key in ["question", "answer"]) or not any(key in body_json for key in ["question", "answer"]), "Dictionary must contain both 'question' and 'answer' keys, or neither."

            # Check that if there is a message, there is no question or answer
            if "message" in body_json:
                assert not any(key in body_json for key in ["question", "answer"]), "Dictionary must not contain 'message' key if it also contains 'question' or 'answer' keys."
            if "question" in body_json:
                assert "message" not in body_json, "Dictionary must not contain 'message' key if it also contains 'question' key."

            # If something else is in the message, raise an assertion
            assert all(key in ["question", "answer", "message"] for key in body_json), "Dictionary must only contain 'question', 'answer', or 'message' keys."

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