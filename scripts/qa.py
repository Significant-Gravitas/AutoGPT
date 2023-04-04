from typing import Dict, Optional
import pika
import json
from dataclasses import dataclass
from colorama import Fore, init
import time
import os


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

    def receive_user_response(self) -> Optional[Dict[str, str]]:
        """Checks to see if there has yet been a single response from the user and if so returns it as a JSON string."""
        # Try to get a message from the queue
        method_frame, _, body = self.channel.basic_get(queue='togpt')

        if method_frame:
            # Acknowledge the message
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            # Return the response as a JSON string
            return json.loads(body.decode())

        return None

class QAClient:
    """The model used by the user to get questions from the Auto GPT Instance and answer them."""

    def __init__(self):
        self.channel = connect_rabbitmq()
        self._previous = None

    def receive(self) -> str:
        # Block and wait until a message/question is received
        method_frame, _, body = self.channel.basic_get(queue='touser', auto_ack=True)

        while not method_frame:
            time.sleep(1)  # You can adjust the sleep duration as needed
            method_frame, _, body = self.channel.basic_get(queue='touser', auto_ack=True)

        self._previous = body.decode()
        return self._previous

    def send_response(self, answer: str) -> None:
        # Send the response to the Auto GPT Instance
        response = {
            "question": self._previous,
            "answer": answer
        }
        self.channel.basic_publish(exchange='', routing_key='togpt', body=json.dumps(response))

    def print_waiting_for_question(self) -> None:
        # Print a message to the user indicating that the Auto GPT Instance is waiting for a question
        print(Fore.GREEN + "Waiting for question...")


# The main loop for the client if you run this script directly
if __name__ == "__main__":
    init(autoreset=True)
    qa_client = QAClient()
    while True:
        print(Fore.GREEN + "Waiting for question...")
        json_received = qa_client.receive()
        if 'question' in json_received:
            question = json.loads(json_received)['question']
            print(Fore.YELLOW + "QUESTION: " + question)
            answer = input(Fore.GREEN + "ANSWER: ")
            qa_client.send_response(answer)
        elif 'message' in json_received:
            message = json.loads(json_received)['message']
            print(Fore.YELLOW + "MESSAGE: " + message)