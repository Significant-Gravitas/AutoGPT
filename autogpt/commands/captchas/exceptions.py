from typing import Optional, Sequence


class ChallengeException(Exception):
    def __init__(
        self, msg: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        self.msg = msg
        self.stacktrace = stacktrace
        super().__init__()

    def __str__(self) -> str:
        exception_msg = "Message: {}\n".format(self.msg)
        if self.stacktrace:
            stacktrace = "\n".join(self.stacktrace)
            exception_msg += "Stacktrace:\n{}".format(stacktrace)
        return exception_msg


class ChallengeTimeoutException(ChallengeException):
    """Step timeout in challenge"""


class RiskControlSystemArmor(ChallengeException):
    """RiskControlSystemArmor"""


class AntiBreakOffWarning(ChallengeException):
    """
    When switching to voiceprint verification exception, it is thrown.
    At this time, the verification has been passed when the checkbox is activated,
    and voiceprint recognition is not required.
    """


class ElementLocationException(ChallengeException):
    """Failure of Strong Location Method Caused by Multilingual Problem"""


class LabelNotFoundException(ChallengeException):
    """Get the challenge label of the exception."""
