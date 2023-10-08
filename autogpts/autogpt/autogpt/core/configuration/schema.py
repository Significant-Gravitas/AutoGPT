import abc
import typing
from typing import Any, Generic, TypeVar
import logging

from pydantic import BaseModel, Field


def UserConfigurable(*args, **kwargs):
    return Field(*args, **kwargs, user_configurable=True)
    # TODO: use this to auto-generate docs for the application configuration


class SystemConfiguration(BaseModel):

    class Config:
        extra = "forbid"
        use_enum_values = True


class SystemSettings(BaseModel):
    pass

S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    class SystemSettings(SystemSettings):
        """A base class for all system settings."""

        name: str
        description: str

        class Config:
            extra = "allow"
            use_enum_values = True


    def __init__(self, settings: S, logger: logging.Logger):
        self._settings = settings
        self._configuration = settings.configuration
        self._logger = logger
