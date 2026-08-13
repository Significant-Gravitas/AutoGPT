import re
from enum import Enum
from typing import Mapping


class BlockError(Exception):
    """An error occurred during the running of a block.

    Exposes the underlying message as ``str(exc)`` without a framing prefix
    so ``yield "error", str(exc)`` surfaces the actual cause to the user.
    The block name and id are kept as attributes for structured logging.
    """

    def __init__(self, message: str, block_name: str, block_id: str) -> None:
        super().__init__(message)
        self.block_name = block_name
        self.block_id = block_id


class BlockInputError(BlockError, ValueError):
    """The block had incorrect inputs, resulting in an error condition"""


class BlockOutputError(BlockError, ValueError):
    """The block had incorrect outputs, resulting in an error condition"""


class BlockExecutionError(BlockError, ValueError):
    """The block failed to execute at runtime, resulting in a handled error"""

    def __init__(self, message: str | None, block_name: str, block_id: str) -> None:
        if message is None:
            message = "Output error was None"
        super().__init__(message, block_name, block_id)


class BlockUnknownError(BlockError):
    """Critical unknown error with block handling"""

    def __init__(self, message: str | None, block_name: str, block_id: str) -> None:
        if not message:
            message = "Unknown error occurred"
        super().__init__(message, block_name, block_id)


class MissingConfigError(Exception):
    """The attempted operation requires configuration which is not available"""


class NotFoundError(ValueError):
    """The requested record was not found, resulting in an error condition"""


class GraphNotFoundError(ValueError):
    """The requested Agent Graph was not found, resulting in an error condition"""


class NeedConfirmation(Exception):
    """The user must explicitly confirm that they want to proceed"""


class NotAuthorizedError(ValueError):
    """The user is not authorized to perform the requested operation"""


class GraphNotAccessibleError(NotAuthorizedError):
    """Raised when attempting to execute a graph that is not accessible to the user."""


class GraphNotInLibraryError(GraphNotAccessibleError):
    """Raised when attempting to execute a graph that is not / no longer in the user's library."""


class PreconditionFailed(Exception):
    """The user must do something else first before trying the current operation"""


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


class ExecutionFailureReason(str, Enum):
    """Structured reasons for terminal graph execution failures."""

    INSUFFICIENT_BALANCE = "insufficient_balance"


_LEGACY_INSUFFICIENT_BALANCE_PATTERNS = (
    re.compile(r"You have no credits left to run an agent\."),
    re.compile(
        r"Insufficient balance of \$-?\d+(?:\.\d+)?, "
        r"where this will cost \$-?\d+(?:\.\d+)?"
    ),
    re.compile(
        r"Insufficient balance to run .+: "
        r"dynamic-cost blocks require a positive balance\."
    ),
    re.compile(r"Organization has -?\d+ credits but needs \d+"),
)


def get_execution_failure_reason(
    error: BaseException | str | None,
    *,
    allow_legacy_text: bool = False,
) -> ExecutionFailureReason | None:
    """Classify trusted exceptions, with an opt-in fallback for persisted text."""
    if isinstance(error, InsufficientBalanceError):
        return ExecutionFailureReason.INSUFFICIENT_BALANCE
    if (
        allow_legacy_text
        and isinstance(error, str)
        and any(
            pattern.fullmatch(error)
            for pattern in _LEGACY_INSUFFICIENT_BALANCE_PATTERNS
        )
    ):
        return ExecutionFailureReason.INSUFFICIENT_BALANCE
    return None


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


class InvalidInputError(ValueError):
    """Raised when user input validation fails (e.g., search term too long)"""

    pass


class DatabaseError(Exception):
    """Raised when there is an error interacting with the database"""

    pass


class RedisError(Exception):
    """Raised when there is an error interacting with Redis"""

    pass


class LinkAlreadyExistsError(ValueError):
    """A platform_linking target (server or user) is already linked."""


class LinkTokenExpiredError(ValueError):
    """A platform_linking token has expired or been consumed."""


class LinkFlowMismatchError(ValueError):
    """A platform_linking token was used for the wrong flow (server vs user)."""


class DuplicateChatMessageError(ValueError):
    """The same user message is already in flight for this chat session."""


class WebhookRegistrationError(Exception):
    """Registering a webhook with an external service failed."""


class ExpertRunPausedError(ValueError):
    """An expert-attributed scheduled/triggered run was refused because the
    expert's schedules are paused (weekly credit budget reached or archive).
    Chat-initiated runs are never gated by this."""

    def __init__(self, message: str, expert_id: str):
        super().__init__(message)
        # args carries both values so the RPC layer can reconstruct the
        # exception; __str__ keeps user-facing rendering to the message.
        self.args = (message, expert_id)
        self.message = message
        self.expert_id = expert_id

    def __str__(self):
        return self.message
