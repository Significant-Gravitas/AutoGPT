import abc
import datetime
import enum
import logging
import uuid
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field


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
        logging.Logger(__name__).warning(f"{__qualname__}.json()")
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
        result = self.dict(**dumps_kwargs)
        encoders = self.Config.json_encoders
        for key, value in result.items():
            for type_, encoder in encoders.items():
                if isinstance(value, type_):
                    result[key] = encoder(value)
        return result

    def json(self, **dumps_kwargs: Any) -> str:
        logging.Logger(__name__).warning(f"{__qualname__}.json()")
        return super().json(**dumps_kwargs)

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
        logging.Logger(__name__).warning(f"{__qualname__}.json()")
        return super().json(**dumps_kwargs)

    def __init__(self, settings: S, logger: logging.Logger):
        self._settings = settings
        self._configuration = settings.configuration
        self._logger = logger


class AFAASMessageType(str, enum.Enum):
    AGENT_LLM = "agent_llm"
    AGENT_AGENT = "agent_agent"
    AGENT_USER = "agent_user"
