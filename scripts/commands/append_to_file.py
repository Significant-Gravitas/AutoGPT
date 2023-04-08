from operator import itemgetter
from file_operations import append_to_file
from commands import Command

class Append_To_File(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        file, text = itemgetter('file', 'text')(kwargs)

        return append_to_file(file, text)