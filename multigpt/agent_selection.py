from enum import Enum, auto


class AgentSelection(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()
    SMART_SELECTION = auto()
