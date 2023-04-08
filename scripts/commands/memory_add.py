from operator import itemgetter

from commands import Command
from memory import PineconeMemory

class Memory_Add(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        string = itemgetter('string')(kwargs)
        memory = PineconeMemory()

        return memory.add(string)
