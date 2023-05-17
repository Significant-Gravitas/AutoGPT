class HandledException(Exception):
    """A custom exception that should print a user-friendly message"""

    def __init__(self, message, *args):
        self.message = message


class CriticalException(HandledException):
    """A custom exception that occurs when execution cannot continue"""

    def __init__(self, message, *args):
        super().__init__(message, *args)
