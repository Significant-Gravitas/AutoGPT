"""Errors for a connection made from a provider sign-in's tokens.

Google and Microsoft sign-in can ask for the data scopes too, and the
frontend then hands the tokens to the backend
(``api/features/integrations/sign_in_connection.py``).
"""


class SignInConnectionError(ValueError):
    """The sign-in's tokens can't become a connection."""


class SignInClientMismatch(SignInConnectionError):
    """Login and connections use different OAuth clients. A provider renews a
    refresh token only for the client it was issued to, so a connection made
    from login's tokens would stop working once its access token expired."""
