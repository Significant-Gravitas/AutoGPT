import openai


class OpenAIError(Exception):
    def __init__(
        self,
        message=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
        code=None,
    ):
        super(OpenAIError, self).__init__(message)

        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except BaseException:
                http_body = (
                    "<Could not decode body as utf-8. "
                    "Please report to support@openai.com>"
                )

        self._message = message
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.code = code
        self.request_id = self.headers.get("request-id", None)
        self.error = self.construct_error_object()
        self.organization = self.headers.get("openai-organization", None)

    def __str__(self):
        msg = self._message or "<empty message>"
        if self.request_id is not None:
            return "Request {0}: {1}".format(self.request_id, msg)
        else:
            return msg

    # Returns the underlying `Exception` (base class) message, which is usually
    # the raw message returned by OpenAI's API. This was previously available
    # in python2 via `error.message`. Unlike `str(error)`, it omits "Request
    # req_..." from the beginning of the string.
    @property
    def user_message(self):
        return self._message

    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self._message,
            self.http_status,
            self.request_id,
        )

    def construct_error_object(self):
        if (
            self.json_body is None
            or "error" not in self.json_body
            or not isinstance(self.json_body["error"], dict)
        ):
            return None

        return openai.api_resources.error_object.ErrorObject.construct_from(
            self.json_body["error"]
        )


class APIError(OpenAIError):
    pass


class TryAgain(OpenAIError):
    pass


class Timeout(OpenAIError):
    pass


class APIConnectionError(OpenAIError):
    def __init__(
        self,
        message,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
        code=None,
        should_retry=False,
    ):
        super(APIConnectionError, self).__init__(
            message, http_body, http_status, json_body, headers, code
        )
        self.should_retry = should_retry


class InvalidRequestError(OpenAIError):
    def __init__(
        self,
        message,
        param,
        code=None,
        http_body=None,
        http_status=None,
        json_body=None,
        headers=None,
    ):
        super(InvalidRequestError, self).__init__(
            message, http_body, http_status, json_body, headers, code
        )
        self.param = param

    def __repr__(self):
        return "%s(message=%r, param=%r, code=%r, http_status=%r, " "request_id=%r)" % (
            self.__class__.__name__,
            self._message,
            self.param,
            self.code,
            self.http_status,
            self.request_id,
        )

    def __reduce__(self):
        return type(self), (
            self._message,
            self.param,
            self.code,
            self.http_body,
            self.http_status,
            self.json_body,
            self.headers,
        )


class AuthenticationError(OpenAIError):
    pass


class PermissionError(OpenAIError):
    pass


class RateLimitError(OpenAIError):
    pass


class ServiceUnavailableError(OpenAIError):
    pass


class InvalidAPIType(OpenAIError):
    pass


class SignatureVerificationError(OpenAIError):
    def __init__(self, message, sig_header, http_body=None):
        super(SignatureVerificationError, self).__init__(message, http_body)
        self.sig_header = sig_header

    def __reduce__(self):
        return type(self), (
            self._message,
            self.sig_header,
            self.http_body,
        )
