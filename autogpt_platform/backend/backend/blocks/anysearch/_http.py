from uuid import UUID

from backend.util.request import HTTPClientError, HTTPServerError, Response


class AnySearchHTTPClientError(HTTPClientError):
    def __init__(self, status_code: int, request_id: str):
        super().__init__(_error_message(status_code, request_id), status_code)
        self.request_id = request_id


class AnySearchHTTPServerError(HTTPServerError):
    def __init__(self, status_code: int, request_id: str):
        super().__init__(_error_message(status_code, request_id), status_code)
        self.request_id = request_id


def raise_for_status(response: Response) -> None:
    if 400 <= response.status <= 499:
        raise AnySearchHTTPClientError(response.status, _request_id(response))
    if 500 <= response.status <= 599:
        raise AnySearchHTTPServerError(response.status, _request_id(response))


def _error_message(status_code: int, request_id: str) -> str:
    return (
        f"AnySearch request failed with HTTP {status_code}: "
        "upstream returned an error (details withheld) "
        f"(request_id: {request_id})"
    )


def _request_id(response: Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        request_id = _validated_request_id(body.get("request_id"))
        if request_id:
            return request_id
    return _validated_request_id(response.headers.get("X-Request-ID"))


def _validated_request_id(value: object) -> str:
    """Only expose the documented UUID v4 correlation ID from error responses."""
    if not isinstance(value, str):
        return ""
    try:
        request_id = UUID(value)
    except ValueError:
        return ""
    return str(request_id) if request_id.version == 4 else ""
