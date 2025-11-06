from typing import Mapping


class MissingConfigError(Exception):
    """The attempted operation requires configuration which is not available"""


class NotFoundError(ValueError):
    """The requested record was not found, resulting in an error condition"""


class NeedConfirmation(Exception):
    """The user must explicitly confirm that they want to proceed"""


class NotAuthorizedError(ValueError):
    """The user is not authorized to perform the requested operation"""


class GraphNotAccessibleError(NotAuthorizedError):
    """Raised when attempting to execute a graph that is not accessible to the user."""


class GraphNotInLibraryError(GraphNotAccessibleError):
    """Raised when attempting to execute a graph that is not / no longer in the user's library."""


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
    content_id: str | None

    def __init__(
        self,
        message: str,
        user_id: str,
        graph_exec_id: str,
        moderation_type: str = "content",
        content_id: str | None = None,
    ):
        super().__init__(message)
        self.args = (message, user_id, graph_exec_id, moderation_type, content_id)
        self.message = message
        self.user_id = user_id
        self.graph_exec_id = graph_exec_id
        self.moderation_type = moderation_type
        self.content_id = content_id

    def __str__(self):
        """Used to display the error message in the frontend, because we str() the error when sending the execution update"""
        if self.content_id:
            return f"{self.message} (Moderation ID: {self.content_id})"
        return self.message


class GraphValidationError(ValueError):
    """Structured validation error for graph validation failures"""

    def __init__(
        self, message: str, node_errors: Mapping[str, Mapping[str, str]] | None = None
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


class DatabaseError(Exception):
    """Raised when there is an error interacting with the database"""

    pass


class RedisError(Exception):
    """Raised when there is an error interacting with Redis"""

    pass
