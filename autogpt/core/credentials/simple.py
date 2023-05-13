import logging
from typing import Type

from autogpt.core.configuration.base import Configuration
from autogpt.core.credentials.base import CredentialsConsumer, CredentialsService


class SimpleCredentialsService(CredentialsService):
    # Credentials defaults are set by classes that inherit from CredentialsConsumer.
    configuration_defaults = {"credentials": {}}

    def __init__(self, configuration: Configuration, logger: logging.Logger):
        self._configuration = configuration.credentials.copy()
        self._logger = logger
        self._credentials = {}

    def register_credentials(
        self,
        service_name: str,
        credentials_consumer: Type[CredentialsConsumer],
    ) -> None:
        default_credentials = credentials_consumer.credentials_defaults
        override_credentials = self._configuration.get(service_name, {})
        credentials = {**default_credentials, **override_credentials}
        self._credentials[service_name] = credentials

    def get_credentials(self, service_name: str) -> dict:
        return self._credentials[service_name].copy()

    def __repr__(self):
        return f"SimpleCredentialsService({list(self._credentials)})"
