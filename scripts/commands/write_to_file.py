from operator import itemgetter
from file_operations import write_to_file
from commands import Command

class Write_To_File(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        file, text = itemgetter('file', 'text')(kwargs)

        return write_to_file(file, text)