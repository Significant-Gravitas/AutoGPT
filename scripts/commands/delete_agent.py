from operator import itemgetter

from commands import Command
import agent_manager as agents

class Delete_Agent(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        key = itemgetter('key')(kwargs)

        result = agents.delete_agent(key)
        
        if not result:
            return f"Agent {key} does not exist."
        
        return f"Agent {key} deleted."