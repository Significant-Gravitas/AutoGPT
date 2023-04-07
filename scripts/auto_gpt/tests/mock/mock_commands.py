from commands import Command, command

class TestCommand(Command):
    def __init__(self):
        super().__init__(name='class_based', description='Class-based test command')

    def __call__(self, arg1: int, arg2: str) -> str:
        return f'{arg1} - {arg2}'

@command('function_based', 'Function-based test command')
def function_based(arg1: int, arg2: str) -> str:
    return f'{arg1} - {arg2}'
