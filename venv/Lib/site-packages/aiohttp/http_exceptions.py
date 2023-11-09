"""Low-level http related exceptions."""


from typing import Optional, Union

from .typedefs import _CIMultiDict

__all__ = ("HttpProcessingError",)


class HttpProcessingError(Exception):
    """HTTP error.

    Shortcut for raising HTTP errors with custom code, message and headers.

    code: HTTP Error code.
    message: (optional) Error message.
    headers: (optional) Headers to be sent in response, a list of pairs
    """

    code = 0
    message = ""
    headers = None

    def __init__(
        self,
        *,
        code: Optional[int] = None,
        message: str = "",
        headers: Optional[_CIMultiDict] = None,
    ) -> None:
        if code is not None:
            self.code = code
        self.headers = headers
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}, message={self.message!r}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self}>"


class BadHttpMessage(HttpProcessingError):

    code = 400
    message = "Bad Request"

    def __init__(self, message: str, *, headers: Optional[_CIMultiDict] = None) -> None:
        super().__init__(message=message, headers=headers)
        self.args = (message,)


class HttpBadRequest(BadHttpMessage):

    code = 400
    message = "Bad Request"


class PayloadEncodingError(BadHttpMessage):
    """Base class for payload errors"""


class ContentEncodingError(PayloadEncodingError):
    """Content encoding error."""


class TransferEncodingError(PayloadEncodingError):
    """transfer encoding error."""


class ContentLengthError(PayloadEncodingError):
    """Not enough data for satisfy content length header."""


class LineTooLong(BadHttpMessage):
    def __init__(
        self, line: str, limit: str = "Unknown", actual_size: str = "Unknown"
    ) -> None:
        super().__init__(
            f"Got more than {limit} bytes ({actual_size}) when reading {line}."
        )
        self.args = (line, limit, actual_size)


class InvalidHeader(BadHttpMessage):
    def __init__(self, hdr: Union[bytes, str]) -> None:
        if isinstance(hdr, bytes):
            hdr = hdr.decode("utf-8", "surrogateescape")
        super().__init__(f"Invalid HTTP Header: {hdr}")
        self.hdr = hdr
        self.args = (hdr,)


class BadStatusLine(BadHttpMessage):
    def __init__(self, line: str = "") -> None:
        if not isinstance(line, str):
            line = repr(line)
        super().__init__(f"Bad status line {line!r}")
        self.args = (line,)
        self.line = line


class InvalidURLError(BadHttpMessage):
    pass
