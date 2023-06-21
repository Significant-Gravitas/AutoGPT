from typing import Any


@dataclass
class CommandFunction:
    """Represents a "function" in OpenAI, which is mapped to a Command in Auto-GPT"""

    name: str
    description: str
    parameters: dict[str, Any]
