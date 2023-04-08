from operator import itemgetter
from commands import Command
import ai_functions as ai

class Evaluate_Code(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again"
        code = itemgetter('code')(kwargs)

        return ai.evaluate_code(code)