from dataclasses import dataclass
from typing import Any, Optional


@dataclass(repr=True)
class CommandError:
    command: Optional[str]
    arguments: dict[str, Any]
    message: str
