class MissingConfigError(Exception):
    """The attempted operation requires configuration which is not available"""


class NeedConfirmation(Exception):
    """The user must explicitly confirm that they want to proceed"""


class InsufficientBalanceError(ValueError):
    user_id: str
    message: str
    balance: float
    amount: float

    def __init__(self, message: str, user_id: str, balance: float, amount: float):
        super().__init__(message)
        self.args = (message, user_id, balance, amount)
        self.message = message
        self.user_id = user_id
        self.balance = balance
        self.amount = amount

    def __str__(self):
        """Used to display the error message in the frontend, because we str() the error when sending the execution update"""
        return self.message
