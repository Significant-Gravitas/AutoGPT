import datetime
import functools
import logging
import os
import re
from collections import namedtuple
from typing import Any, Callable, Dict, Iterable, List, Tuple  # noqa

from .abc import AbstractAccessLogger
from .web_request import BaseRequest
from .web_response import StreamResponse

KeyMethod = namedtuple("KeyMethod", "key method")


class AccessLogger(AbstractAccessLogger):
    """Helper object to log access.

    Usage:
        log = logging.getLogger("spam")
        log_format = "%a %{User-Agent}i"
        access_logger = AccessLogger(log, log_format)
        access_logger.log(request, response, time)

    Format:
        %%  The percent sign
        %a  Remote IP-address (IP-address of proxy if using reverse proxy)
        %t  Time when the request was started to process
        %P  The process ID of the child that serviced the request
        %r  First line of request
        %s  Response status code
        %b  Size of response in bytes, including HTTP headers
        %T  Time taken to serve the request, in seconds
        %Tf Time taken to serve the request, in seconds with floating fraction
            in .06f format
        %D  Time taken to serve the request, in microseconds
        %{FOO}i  request.headers['FOO']
        %{FOO}o  response.headers['FOO']
        %{FOO}e  os.environ['FOO']

    """

    LOG_FORMAT_MAP = {
        "a": "remote_address",
        "t": "request_start_time",
        "P": "process_id",
        "r": "first_request_line",
        "s": "response_status",
        "b": "response_size",
        "T": "request_time",
        "Tf": "request_time_frac",
        "D": "request_time_micro",
        "i": "request_header",
        "o": "response_header",
    }

    LOG_FORMAT = '%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    FORMAT_RE = re.compile(r"%(\{([A-Za-z0-9\-_]+)\}([ioe])|[atPrsbOD]|Tf?)")
    CLEANUP_RE = re.compile(r"(%[^s])")
    _FORMAT_CACHE: Dict[str, Tuple[str, List[KeyMethod]]] = {}

    def __init__(self, logger: logging.Logger, log_format: str = LOG_FORMAT) -> None:
        """Initialise the logger.

        logger is a logger object to be used for logging.
        log_format is a string with apache compatible log format description.

        """
        super().__init__(logger, log_format=log_format)

        _compiled_format = AccessLogger._FORMAT_CACHE.get(log_format)
        if not _compiled_format:
            _compiled_format = self.compile_format(log_format)
            AccessLogger._FORMAT_CACHE[log_format] = _compiled_format

        self._log_format, self._methods = _compiled_format

    def compile_format(self, log_format: str) -> Tuple[str, List[KeyMethod]]:
        """Translate log_format into form usable by modulo formatting

        All known atoms will be replaced with %s
        Also methods for formatting of those atoms will be added to
        _methods in appropriate order

        For example we have log_format = "%a %t"
        This format will be translated to "%s %s"
        Also contents of _methods will be
        [self._format_a, self._format_t]
        These method will be called and results will be passed
        to translated string format.

        Each _format_* method receive 'args' which is list of arguments
        given to self.log

        Exceptions are _format_e, _format_i and _format_o methods which
        also receive key name (by functools.partial)

        """
        # list of (key, method) tuples, we don't use an OrderedDict as users
        # can repeat the same key more than once
        methods = list()

        for atom in self.FORMAT_RE.findall(log_format):
            if atom[1] == "":
                format_key1 = self.LOG_FORMAT_MAP[atom[0]]
                m = getattr(AccessLogger, "_format_%s" % atom[0])
                key_method = KeyMethod(format_key1, m)
            else:
                format_key2 = (self.LOG_FORMAT_MAP[atom[2]], atom[1])
                m = getattr(AccessLogger, "_format_%s" % atom[2])
                key_method = KeyMethod(format_key2, functools.partial(m, atom[1]))

            methods.append(key_method)

        log_format = self.FORMAT_RE.sub(r"%s", log_format)
        log_format = self.CLEANUP_RE.sub(r"%\1", log_format)
        return log_format, methods

    @staticmethod
    def _format_i(
        key: str, request: BaseRequest, response: StreamResponse, time: float
    ) -> str:
        if request is None:
            return "(no headers)"

        # suboptimal, make istr(key) once
        return request.headers.get(key, "-")

    @staticmethod
    def _format_o(
        key: str, request: BaseRequest, response: StreamResponse, time: float
    ) -> str:
        # suboptimal, make istr(key) once
        return response.headers.get(key, "-")

    @staticmethod
    def _format_a(request: BaseRequest, response: StreamResponse, time: float) -> str:
        if request is None:
            return "-"
        ip = request.remote
        return ip if ip is not None else "-"

    @staticmethod
    def _format_t(request: BaseRequest, response: StreamResponse, time: float) -> str:
        now = datetime.datetime.utcnow()
        start_time = now - datetime.timedelta(seconds=time)
        return start_time.strftime("[%d/%b/%Y:%H:%M:%S +0000]")

    @staticmethod
    def _format_P(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return "<%s>" % os.getpid()

    @staticmethod
    def _format_r(request: BaseRequest, response: StreamResponse, time: float) -> str:
        if request is None:
            return "-"
        return "{} {} HTTP/{}.{}".format(
            request.method,
            request.path_qs,
            request.version.major,
            request.version.minor,
        )

    @staticmethod
    def _format_s(request: BaseRequest, response: StreamResponse, time: float) -> int:
        return response.status

    @staticmethod
    def _format_b(request: BaseRequest, response: StreamResponse, time: float) -> int:
        return response.body_length

    @staticmethod
    def _format_T(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return str(round(time))

    @staticmethod
    def _format_Tf(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return "%06f" % time

    @staticmethod
    def _format_D(request: BaseRequest, response: StreamResponse, time: float) -> str:
        return str(round(time * 1000000))

    def _format_line(
        self, request: BaseRequest, response: StreamResponse, time: float
    ) -> Iterable[Tuple[str, Callable[[BaseRequest, StreamResponse, float], str]]]:
        return [(key, method(request, response, time)) for key, method in self._methods]

    def log(self, request: BaseRequest, response: StreamResponse, time: float) -> None:
        try:
            fmt_info = self._format_line(request, response, time)

            values = list()
            extra = dict()
            for key, value in fmt_info:
                values.append(value)

                if key.__class__ is str:
                    extra[key] = value
                else:
                    k1, k2 = key  # type: ignore[misc]
                    dct = extra.get(k1, {})  # type: ignore[var-annotated,has-type]
                    dct[k2] = value  # type: ignore[index,has-type]
                    extra[k1] = dct  # type: ignore[has-type,assignment]

            self.logger.info(self._log_format % tuple(values), extra=extra)
        except Exception:
            self.logger.exception("Error in logging")
