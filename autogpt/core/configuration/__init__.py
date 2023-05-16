"""The configuration encapsulates settings for all Agent subsystems."""
from autogpt.core.configuration.schema import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Next up is creating a basic implementation.\n"
        "5/14: Use pydantic to set up core models for system configuration and credentials.\n"
        "5/16: Configuration is working. It can be compiled across the system. User overrides are working.\n"
        "      Credentials are appropriately hidden. This is effectively done unless I've missed a design constraint.\n"
    ),
)
