from random import uniform
from time import monotonic, sleep
from typing import Any

from requests import Response, Session
from requests.exceptions import ConnectionError, RequestException, SSLError
from requests.exceptions import Timeout as RequestTimeout
from urllib3.util import Timeout

_MAX_ATTEMPTS = 3


def create_trace_export_session() -> Session:
    """Retry OTLP requests within a window of at most three request timeouts."""
    return _TraceExportSession()


class _TraceExportSession(Session):
    def post(
        self,
        url: str | bytes,
        data: Any = None,
        json: Any = None,
        **kwargs: Any,
    ) -> Response:
        request_timeout = float(kwargs["timeout"])
        deadline = monotonic() + request_timeout * _MAX_ATTEMPTS
        last_error: RequestException | None = None
        for attempt in range(_MAX_ATTEMPTS):
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            kwargs["timeout"] = Timeout(total=min(request_timeout, remaining))
            try:
                return super().post(url, data=data, json=json, **kwargs)
            except SSLError as error:
                raise RequestException("OTLP trace export TLS failure") from error
            except (ConnectionError, RequestTimeout) as error:
                last_error = error
            remaining = deadline - monotonic()
            if attempt + 1 < _MAX_ATTEMPTS and remaining > 0:
                sleep(uniform(0, min(0.1 * 2**attempt, remaining / 2)))

        # ConnectionError would make the OTLP exporter retry the exhausted batch again.
        raise RequestException(
            "OTLP trace export transport retry budget exhausted"
        ) from last_error
