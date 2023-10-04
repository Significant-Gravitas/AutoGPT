import abc
import typing
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field


def UserConfigurable(*args, **kwargs):
    return Field(*args, **kwargs, user_configurable=True)


class SystemConfiguration(BaseModel):
    def get_user_config(self) -> dict[str, Any]:
        return _get_user_config_fields(self)

    class Config:
        extra = "forbid"
        use_enum_values = True


class SystemSettings(BaseModel):
    """A base class for all system settings."""

    name: str
    description: str

    class Config:
        extra = "forbid"
        use_enum_values = True


S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    default_settings: typing.ClassVar[S]

    @classmethod
    def get_user_config(cls) -> dict[str, Any]:
        return _get_user_config_fields(cls.default_settings)

    @classmethod
    def build_agent_configuration(cls, configuration: dict) -> S:
        """Process the configuration for this object."""

        defaults = cls.default_settings.dict()
        final_configuration = deep_update(defaults, configuration)

        return cls.default_settings.__class__.parse_obj(final_configuration)


def _get_user_config_fields(instance: BaseModel) -> dict[str, Any]:
    """
    Get the user config fields of a Pydantic model instance.

    Args:
        instance: The Pydantic model instance.

    Returns:
        The user config fields of the instance.
    """
    user_config_fields = {}

    for name, value in instance.__dict__.items():
        field_info = instance.__fields__[name]
        if "user_configurable" in field_info.field_info.extra:
            user_config_fields[name] = value
        elif isinstance(value, SystemConfiguration):
            user_config_fields[name] = value.get_user_config()
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_fields[name] = [i.get_user_config() for i in value]
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_fields[name] = {
                k: v.get_user_config() for k, v in value.items()
            }

    return user_config_fields


def deep_update(original_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Args:
        original_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary to update with.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            original_dict[key] = deep_update(original_dict[key], value)
        else:
            original_dict[key] = value
    return original_dict
