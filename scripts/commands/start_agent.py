from operator import itemgetter

from config import Config
from commands import Command, CommandManager
import agent_manager as agents
import speak

cfg = Config()

class Start_Agent(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        name, task, prompt = itemgetter('name', 'task', 'prompt')(kwargs)
        model = cfg.fast_llm_model
        
        # Remove underscores from name
        voice_name = name.replace("_", " ")

        first_message = f"""You are {name}.  Respond with: "Acknowledged"."""
        agent_intro = f"{voice_name} here, Reporting for duty!"

        # Create agent
        if cfg.speak_mode:
            speak.say_text(agent_intro, 1)
        key, ack = agents.create_agent(task, first_message, model)

        if cfg.speak_mode:
            speak.say_text(f"Hello {voice_name}. Your task is as follows. {task}.")

        # Assign task (prompt), get response
        message_agent = CommandManager.get('message_agent')
        agent_response = message_agent.execute(key=key, prompt=prompt)

        return f"Agent {name} created with key {key}. First response: {agent_response}"
