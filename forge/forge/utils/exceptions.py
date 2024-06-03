import inspect
import sys
import traceback
from typing import Optional


def get_exception_message():
    """Get current exception type and message."""
    exc_type, exc_value, _ = sys.exc_info()
    exception_message = f"{exc_type.__name__}: {exc_value}" if exc_type else exc_value
    return exception_message


def get_detailed_traceback():
    """Get current exception traceback with local variables."""
    _, _, exc_tb = sys.exc_info()
    detailed_traceback = "Traceback (most recent call last):\n"
    formatted_tb = traceback.format_tb(exc_tb)
    detailed_traceback += "".join(formatted_tb)

    # Optionally add local variables to the traceback information
    detailed_traceback += "\nLocal variables by frame, innermost last:\n"
    while exc_tb:
        frame = exc_tb.tb_frame
        lineno = exc_tb.tb_lineno
        function_name = frame.f_code.co_name

        # Format frame information
        detailed_traceback += (
            f"  Frame {function_name} in {frame.f_code.co_filename} at line {lineno}\n"
        )

        # Get local variables for the frame
        local_vars = inspect.getargvalues(frame).locals
        for var_name, value in local_vars.items():
            detailed_traceback += f"    {var_name} = {value}\n"

        exc_tb = exc_tb.tb_next

    return detailed_traceback


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


class AgentTerminated(AgentException):
    """The agent terminated or was terminated"""


class AgentFinished(AgentTerminated):
    """The agent self-terminated"""


class ConfigurationError(AgentException):
    """Error caused by invalid, incompatible or otherwise incorrect configuration"""


class InvalidAgentResponseError(AgentException):
    """The LLM deviated from the prescribed response format"""


class UnknownCommandError(AgentException):
    """The AI tried to use an unknown command"""

    hint = "Do not try to use this command again."


class CommandExecutionError(AgentException):
    """An error occurred when trying to execute the command"""


class InvalidArgumentError(CommandExecutionError):
    """The command received an invalid argument"""


class OperationNotAllowedError(CommandExecutionError):
    """The agent is not allowed to execute the proposed operation"""


class TooMuchOutputError(CommandExecutionError):
    """The operation generated more output than what the Agent can process"""
