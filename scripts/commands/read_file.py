from operator import itemgetter
from file_operations import read_file
from commands import Command

class Read_File(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        file = itemgetter('file')(kwargs)

        return read_file(file)