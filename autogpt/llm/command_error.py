from dataclasses import dataclass
from typing import Any, Optional


@dataclass(repr=True)
class CommandError:
    command: str
    arguments: dict[str, Any]
    message: str
