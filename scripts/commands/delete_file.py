from operator import itemgetter
from file_operations import delete_file
from commands import Command

class Delete_File(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        file = itemgetter('file')(kwargs)

        return delete_file(file)