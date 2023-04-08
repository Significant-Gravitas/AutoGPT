from operator import itemgetter

from config import Config
from commands import Command
import agent_manager as agents
import speak

cfg = Config()

class Message_Agent(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        key, message = itemgetter('key', 'message')(kwargs)

        # Check if the key is a valid integer
        if (str(key).isnumeric()):
            agent_response = agents.message_agent(key, message)
        else:
            return "Invalid key, must be numeric."

        # Speak response
        if cfg.speak_mode:
            speak.say_text(agent_response, 1)

        return agent_response
