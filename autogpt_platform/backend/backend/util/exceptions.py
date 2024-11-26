class MissingConfigError(Exception):
    """The attempted operation requires configuration which is not available"""


class NeedConfirmation(Exception):
    """The user must explicitly confirm that they want to proceed"""
