class MissingConfigError(Exception):
    """The attempted operation requires configuration which is not available"""


class NotFoundError(ValueError):
    """The requested record was not found, resulting in an error condition"""


class NeedConfirmation(Exception):
    """The user must explicitly confirm that they want to proceed"""


class NotAuthorizedError(ValueError):
    """The user is not authorized to perform the requested operation"""


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


class ModerationError(ValueError):
    """Content moderation failure during execution"""

    user_id: str
    message: str
    graph_exec_id: str
    moderation_type: str

    def __init__(
        self,
        message: str,
        user_id: str,
        graph_exec_id: str,
        moderation_type: str = "content",
    ):
        super().__init__(message)
        self.args = (message, user_id, graph_exec_id, moderation_type)
        self.message = message
        self.user_id = user_id
        self.graph_exec_id = graph_exec_id
        self.moderation_type = moderation_type

    def __str__(self):
        """Used to display the error message in the frontend, because we str() the error when sending the execution update"""
        return self.message


class GraphValidationError(ValueError):
    """Structured validation error for graph validation failures"""

    def __init__(
        self, message: str, node_errors: dict[str, dict[str, str]] | None = None
    ):
        super().__init__(message)
        self.message = message
        self.node_errors = node_errors or {}

    def __str__(self):
        return self.message + "".join(
            [
                f"\n  {node_id}:"
                + "".join([f"\n    {k}: {e}" for k, e in errors.items()])
                for node_id, errors in self.node_errors.items()
            ]
        )
