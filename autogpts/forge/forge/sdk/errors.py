from typing import Optional


class NotFoundError(Exception):
    pass


class AgentException(Exception):
    """Base class for specific exceptions relevant in the execution of Agents"""

    message: str

    hint: Optional[str] = None
    """A hint which can be passed to the LLM to reduce reoccurrence of this error"""

    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(AgentException):
    """Error caused by invalid, incompatible or otherwise incorrect configuration"""


class InvalidAgentResponseError(AgentException):
    """The LLM deviated from the prescribed response format"""


class UnknownCommandError(AgentException):
    """The AI tried to use an unknown command"""

    hint = "Do not try to use this command again."


class DuplicateOperationError(AgentException):
    """The proposed operation has already been executed"""


class CommandExecutionError(AgentException):
    """An error occured when trying to execute the command"""


class InvalidArgumentError(CommandExecutionError):
    """The command received an invalid argument"""


class OperationNotAllowedError(CommandExecutionError):
    """The agent is not allowed to execute the proposed operation"""


class AccessDeniedError(CommandExecutionError):
    """The operation failed because access to a required resource was denied"""


class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""


class TooMuchOutputError(CommandExecutionError):
    """The operation generated more output than what the Agent can process"""
