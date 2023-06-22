import kafka

from autogpt.core.schema import BaseEvent, BaseMessageChannel


class KafkaChannel(BaseMessageChannel):
    def __init__(self, name: str, host: str, port: int, topic: str) -> None:
        """Initializes a Kafka channel."""
        self.name = name
        self.host = host
        self.port = port
        self.topic = topic
        self.producer = kafka.KafkaProducer(bootstrap_servers=f"{host}:{port}")
        self.consumer = kafka.KafkaConsumer(topic, bootstrap_servers=f"{host}:{port}")

    async def get(self) -> BaseEvent:
        """Gets a message from the channel."""
        for msg in self.consumer:
            self.received_message_count += 1
            self.received_bytes_count += len(msg.value)
            return msg.value

    async def send(self, message: BaseEvent) -> None:
        """Sends a message to the channel."""
        self.sent_message_count += 1
        self.send_bytes_count += len(message)
        self.producer.send(self.topic, message)
