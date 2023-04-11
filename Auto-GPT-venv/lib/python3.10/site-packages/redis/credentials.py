from typing import Optional, Tuple, Union


class CredentialProvider:
    """
    Credentials Provider.
    """

    def get_credentials(self) -> Union[Tuple[str], Tuple[str, str]]:
        raise NotImplementedError("get_credentials must be implemented")


class UsernamePasswordCredentialProvider(CredentialProvider):
    """
    Simple implementation of CredentialProvider that just wraps static
    username and password.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = username or ""
        self.password = password or ""

    def get_credentials(self):
        if self.username:
            return self.username, self.password
        return (self.password,)
