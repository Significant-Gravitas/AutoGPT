from scripts.qa import connect_to_redis
from scripts import chat
from typing import List

class QAAgent:
    """The model used by the Auto GPT Instance to ask questions and receive answers from the user."""

    def __init__(self):
        self.redis = connect_to_redis()

    def message_user(self, message: str) -> str:
        """Notify the user of a message and return a message to the gpt agent to check back later for a response."""
        # Send the message to the user
        self.redis.lpush("touser", message)
        return "You have sent the message to the user. You may or may not receive a response. You may ask other questions without waiting for a response. You may also send other messages to the user without waiting for a response. Or, you may use the command \"wait_user\" to wait for a response."

    def receive_all_user_responses(self) -> List[chat.ChatMessage]:
        """Checks to see if there has yet been a single response from the user and if so returns it as a JSON string."""
        out = []

        while self.redis.llen("togpt") > 0:
            message = self.redis.rpop("togpt")
            out.append(chat.create_chat_message("user", message))

        return out

    def wait(self) -> str:
        """Wait for a response from the user and return it as a JSON string."""
        self.redis.lpush("touser", "Waiting for a response from the user...")
        while self.redis.llen("togpt") == 0:
            pass

        return self.redis.rpop("togpt")
