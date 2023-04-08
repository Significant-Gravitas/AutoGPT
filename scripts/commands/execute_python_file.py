from operator import itemgetter
from commands import Command
from execute_code import execute_python_file

class Execute_Python_File(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        file = itemgetter('file')(kwargs)
        
        return execute_python_file(file)