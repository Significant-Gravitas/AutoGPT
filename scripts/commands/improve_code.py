from operator import itemgetter
from commands import Command
import ai_functions as ai

class Improve_Code(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again"
        suggestions, code = itemgetter('suggestions', 'code')(kwargs)

        return ai.improve_code(suggestions, code)