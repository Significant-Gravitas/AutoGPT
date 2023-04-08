from commands import Command

class Task_Complete(Command):
    def __init__(self):
        super().__init__()

    def execute(self, **kwargs):
        print("Shutting down...")
        quit()
