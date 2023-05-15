"""The configuration encapsulates settings for all Agent subsystems."""
from autogpt.core.configuration.base import AgentConfiguration
from autogpt.core.configuration.schema import (
    Credentials,
    Configurable,
    SystemConfiguration,
    SystemSettings,
)

from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Next up is creating a basic implementation."
    ),
)
