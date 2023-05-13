import abc
from typing import Type


class CredentialsConsumer(abc.ABC):
    credentials_defaults = {}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...


class CredentialsService(abc.ABC):
    configuration_defaults = {"credentials": {}}

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def register_credentials(
        self,
        service_name: str,
        credentials_consumer: Type[CredentialsConsumer],
    ) -> None:
        """Add credentials for a service, resolving defaults with user provided overrides.."""
        ...

    @abc.abstractmethod
    def get_credentials(self, service_name: str) -> dict:
        """Get credentials for a service."""
        ...

    @abc.abstractmethod
    def __repr__(self):
        ...
