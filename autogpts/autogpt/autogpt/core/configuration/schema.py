from __future__ import annotations
import abc
import datetime
import enum
import logging
import uuid
from typing import Any, Generic, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, Field



LOG = logging.Logger(__name__)


def UserConfigurable(*args, **kwargs):
    return Field(*args, **kwargs, user_configurable=True)
    # TODO: use this to auto-generate docs for the application configuration


class SystemConfiguration(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True
        json_encoders = {
            uuid.UUID: lambda v: str(v),
            float: lambda v: str(
                9999.99 if v == float("inf") or v == float("-inf") else v
            ),
            datetime.datetime: lambda v: v.isoformat(),
        }
        # This is a list of Field to Exclude during serialization
        default_exclude = {
            "agent",
            "workspace",
            "prompt_manager",
            "chat_model_provider",
            "memory",
            "tool_registry",
            "prompt_settings",
            "systems",
            "configuration",
            "name",
            "description",
        }
        allow_inf_nan = False

    def json(self, **dumps_kwargs: Any) -> str:
        LOG.warning(f"{__qualname__}.json()")
        return super().json(**dumps_kwargs)


class AFAASModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True
        json_encoders = {
            uuid.UUID: lambda v: str(v),
            float: lambda v: str(
                9999.99 if v == float("inf") or v == float("-inf") else v
            ),
            datetime.datetime: lambda v: str(v.isoformat()),
        }
        arbitrary_types_allowed = True
        allow_inf_nan = False
        default_exclude = {}

    # TODO: #21 https://github.com/ph-ausseil/afaas/issues/21
    created_at: datetime.datetime = datetime.datetime.now()
    modified_at: datetime.datetime = datetime.datetime.now()

    def dict_memory(self, **dumps_kwargs: Any) -> dict:
        LOG.debug(f"FIXME: Temporary implementation before a to pydantic 2.0.0")
        result = self.dict(**dumps_kwargs)
        encoders = self.Config.json_encoders
        for key, value in result.items():
            for type_, encoder in encoders.items():
                if isinstance(value, type_):
                    result[key] = encoder(value)
        return result

    
    def dict(self, include_all=False, *args, **kwargs):
        """
        Serialize the object to a dictionary representation.

        Args:
            remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
            *args: Additional positional arguments to pass to the base class's dict method.
            **kwargs: Additional keyword arguments to pass to the base class's dict method.
            kwargs['exclude'] excludes the fields from the serialization

        Returns:
            dict: A dictionary representation of the object.
        """
        self.prepare_values_before_serialization()  # Call the custom treatment before .dict()
        if not include_all:
            kwargs["exclude"] = self.Config.default_exclude
        # Call the .dict() method with the updated exclude_arg
        return super().dict(*args, **kwargs)

    def json(self, *args, **kwargs):
        """
        Serialize the object to a dictionary representation.

        Args:
            remove_technical_values (bool, optional): Whether to exclude technical values. Default is True.
            *args: Additional positional arguments to pass to the base class's dict method.
            **kwargs: Additional keyword arguments to pass to the base class's dict method.
            kwargs['exclude'] excludes the fields from the serialization

        Returns:
            dict: A dictionary representation of the object.
        """
        logging.Logger(__name__).warning(
            "Warning : Recomended use json_api() or json_memory()"
        )
        logging.Logger(__name__).warning("BaseAgent.SystemSettings.json()")
        self.prepare_values_before_serialization()  # Call the custom treatment before .json()
        kwargs["exclude"] = self.Config.default_exclude
        return super().json(*args, **kwargs)

    # TODO Implement a BaseSettings class and move it to the BaseSettings ?
    def prepare_values_before_serialization(self):
            pass

    def __str__(self):
        lines = [f"{self.__class__.__name__}("]

        for field_name, field_value in self.dict().items():
            formatted_field_name = field_name.replace("_", " ").capitalize()
            if field_value is None:
                value_str = "Not provided"
            elif isinstance(field_value, list):
                value_str = ", ".join(map(str, field_value))
                value_str = f"[{value_str}]"
            elif isinstance(field_value, dict):
                value_str = str(field_value)
            else:
                value_str = str(field_value)

            lines.append(f"  {formatted_field_name}: {value_str}")

        lines.append(")")
        return "\n".join(lines)
    
    @staticmethod
    def generate_uuid() :
        return str(uuid.uuid4())


class SystemSettings(AFAASModel):
    class Config(AFAASModel.Config):
        default_exclude = {
            "agent",
            "workspace",
            "prompt_manager",
            "chat_model_provider",
            "memory",
            "tool_registry",
            "prompt_settings",
            "systems",
            "configuration",
            "agent_setting_module",
            "agent_setting_class",
            "name",
            "description",
            "parent_agent",
            "current_task",
        }


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
            allow_inf_nan = False

    def json(self, **dumps_kwargs: Any) -> str:
        LOG.warning(f"{__qualname__}.json()")
        return super().json(**dumps_kwargs)

    def __init__(self, settings: S, logger: logging.Logger):
        self._settings = settings
        self._configuration = settings.configuration
        self._logger = logger


class AFAASMessageType(str, enum.Enum):
    AGENT_LLM = "agent_llm"
    AGENT_AGENT = "agent_agent"
    AGENT_USER = "agent_user"
