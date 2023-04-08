from operator import itemgetter
from file_operations import search_files
from commands import Command

class Search_Files(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        directory = itemgetter('directory')(kwargs)

        return search_files(directory)