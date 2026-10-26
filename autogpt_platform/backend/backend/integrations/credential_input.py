"""Validation for credentials that a user submits through the API.

The credential models in ``backend.data.model`` deliberately carry no
validators: system credentials are built at import time with empty secrets
for unset settings, and stored rows that fail to decode are dropped. So user
input is checked here, at the request boundary, before it is stored.
"""

from pydantic import SecretStr

from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    HostScopedCredentials,
    UserPasswordCredentials,
)


class InvalidCredentialInputError(ValueError):
    """User-supplied credentials are blank or malformed.

    The message never includes the submitted secret.
    """


def normalize_user_supplied_credentials(credentials: Credentials) -> Credentials:
    """Strip and check the secrets of credentials that came from a user request.

    - API key: stripped, and must not be empty. An expiry, if given, must be
      in the future.
    - Username/password: the username is stripped and required. The password is
      left exactly as given, since it may legitimately contain surrounding
      spaces or be empty.
    - Host-scoped: the host is stripped and required, and there must be at least
      one header. Header names and values are stripped and must not be empty.
    - The title, if any, is stripped. A blank title becomes ``None``.

    OAuth credentials are returned unchanged. The object is updated in place and
    returned.

    Raises:
        InvalidCredentialInputError: if a required value is blank, or an API
            key's expiry has already passed.
    """
    if isinstance(credentials, APIKeyCredentials):
        api_key = credentials.api_key.get_secret_value().strip()
        if not api_key:
            raise InvalidCredentialInputError("API key must not be empty")
        credentials.api_key = SecretStr(api_key)
        if credentials.is_expired():
            raise InvalidCredentialInputError("Expiry must be in the future")

    elif isinstance(credentials, UserPasswordCredentials):
        username = credentials.username.get_secret_value().strip()
        if not username:
            raise InvalidCredentialInputError("Username must not be empty")
        credentials.username = SecretStr(username)

    elif isinstance(credentials, HostScopedCredentials):
        host = credentials.host.strip()
        if not host:
            raise InvalidCredentialInputError("Host must not be empty")
        if not credentials.headers:
            raise InvalidCredentialInputError("At least one header is required")
        headers: dict[str, SecretStr] = {}
        for name, value in credentials.headers.items():
            name = name.strip()
            stripped_value = value.get_secret_value().strip()
            if not name:
                raise InvalidCredentialInputError("Header names must not be empty")
            if not stripped_value:
                # Name the header (not its value) so the user can find it.
                raise InvalidCredentialInputError(f"Header '{name}' must have a value")
            if name in headers:
                raise InvalidCredentialInputError(f"Header '{name}' is repeated")
            headers[name] = SecretStr(stripped_value)
        credentials.host = host
        credentials.headers = headers

    if credentials.title is not None:
        credentials.title = credentials.title.strip() or None

    return credentials
