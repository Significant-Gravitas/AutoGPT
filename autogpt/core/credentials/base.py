import abc


class CredentialsManager(abc.ABC):
    configuration_defaults = {"credentials": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def add_credentials(self, service_name: str, credentials: dict) -> None:
        """Add credentials for a service."""
        ...

    @abc.abstractmethod
    def get_credentials(self, service_name: str) -> dict:
        """Get credentials for a service."""
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...
