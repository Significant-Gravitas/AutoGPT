"""The errors the fake Stripe answers with, shaped as Stripe's."""


class ApiError(Exception):
    """A request Stripe would refuse, with the error body it would send."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        kind: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        error = {"type": kind, "message": message, "code": code, "param": param}
        self.status = status
        self.body = {"error": {k: v for k, v in error.items() if v is not None}}


class NotModelled(ApiError):
    """A call to a URL the fake serves, with parameters it doesn't model (a
    price change, a trial end). It is refused like an unknown URL and
    journaled as unserved, so a journey sees the platform reach past the
    fake rather than a made-up answer."""

    def __init__(self, what: str) -> None:
        super().__init__(400, f"The fake Stripe doesn't model {what}.")


def missing(param: str) -> ApiError:
    return ApiError(
        400,
        f"Missing required param: {param}.",
        code="parameter_missing",
        param=param,
    )
