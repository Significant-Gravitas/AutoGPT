from operator import itemgetter

from commands import Command
import agent_manager as agents

class List_Agents(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        return agents.list_agents()
