import requests

_image_not_found_explanation_fragments = frozenset(
    fragment.lower() for fragment in [
        'no such image',
        'not found: does not exist or no pull access',
        'repository does not exist',
        'was found but does not match the specified platform',
    ]
)


class DockerException(Exception):
    """
    A base class from which all other exceptions inherit.

    If you want to catch all errors that the Docker SDK might raise,
    catch this base exception.
    """


def create_api_error_from_http_exception(e):
    """
    Create a suitable APIError from requests.exceptions.HTTPError.
    """
    response = e.response
    try:
        explanation = response.json()['message']
    except ValueError:
        explanation = (response.content or '').strip()
    cls = APIError
    if response.status_code == 404:
        explanation_msg = (explanation or '').lower()
        if any(fragment in explanation_msg
               for fragment in _image_not_found_explanation_fragments):
            cls = ImageNotFound
        else:
            cls = NotFound
    raise cls(e, response=response, explanation=explanation) from e


class APIError(requests.exceptions.HTTPError, DockerException):
    """
    An HTTP error from the API.
    """
    def __init__(self, message, response=None, explanation=None):
        # requests 1.2 supports response as a keyword argument, but
        # requests 1.1 doesn't
        super().__init__(message)
        self.response = response
        self.explanation = explanation

    def __str__(self):
        message = super().__str__()

        if self.is_client_error():
            message = '{} Client Error for {}: {}'.format(
                self.response.status_code, self.response.url,
                self.response.reason)

        elif self.is_server_error():
            message = '{} Server Error for {}: {}'.format(
                self.response.status_code, self.response.url,
                self.response.reason)

        if self.explanation:
            message = f'{message} ("{self.explanation}")'

        return message

    @property
    def status_code(self):
        if self.response is not None:
            return self.response.status_code

    def is_error(self):
        return self.is_client_error() or self.is_server_error()

    def is_client_error(self):
        if self.status_code is None:
            return False
        return 400 <= self.status_code < 500

    def is_server_error(self):
        if self.status_code is None:
            return False
        return 500 <= self.status_code < 600


class NotFound(APIError):
    pass


class ImageNotFound(NotFound):
    pass


class InvalidVersion(DockerException):
    pass


class InvalidRepository(DockerException):
    pass


class InvalidConfigFile(DockerException):
    pass


class InvalidArgument(DockerException):
    pass


class DeprecatedMethod(DockerException):
    pass


class TLSParameterError(DockerException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg + (". TLS configurations should map the Docker CLI "
                           "client configurations. See "
                           "https://docs.docker.com/engine/articles/https/ "
                           "for API details.")


class NullResource(DockerException, ValueError):
    pass


class ContainerError(DockerException):
    """
    Represents a container that has exited with a non-zero exit code.
    """
    def __init__(self, container, exit_status, command, image, stderr):
        self.container = container
        self.exit_status = exit_status
        self.command = command
        self.image = image
        self.stderr = stderr

        err = f": {stderr}" if stderr is not None else ""
        msg = ("Command '{}' in image '{}' returned non-zero exit "
               "status {}{}").format(command, image, exit_status, err)

        super().__init__(msg)


class StreamParseError(RuntimeError):
    def __init__(self, reason):
        self.msg = reason


class BuildError(DockerException):
    def __init__(self, reason, build_log):
        super().__init__(reason)
        self.msg = reason
        self.build_log = build_log


class ImageLoadError(DockerException):
    pass


def create_unexpected_kwargs_error(name, kwargs):
    quoted_kwargs = [f"'{k}'" for k in sorted(kwargs)]
    text = [f"{name}() "]
    if len(quoted_kwargs) == 1:
        text.append("got an unexpected keyword argument ")
    else:
        text.append("got unexpected keyword arguments ")
    text.append(', '.join(quoted_kwargs))
    return TypeError(''.join(text))


class MissingContextParameter(DockerException):
    def __init__(self, param):
        self.param = param

    def __str__(self):
        return (f"missing parameter: {self.param}")


class ContextAlreadyExists(DockerException):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return (f"context {self.name} already exists")


class ContextException(DockerException):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return (self.msg)


class ContextNotFound(DockerException):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return (f"context '{self.name}' not found")
