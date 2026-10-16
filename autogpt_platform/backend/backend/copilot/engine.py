"""Which execution engine runs a platform-route turn.

Lives apart from the processor so the answer can be read without importing
the engines themselves: the processor pulls in both services at module
level, and the API layer needs this decision to describe a connection
before any turn exists.
"""

from typing import Literal

from backend.copilot.config import ChatConfig

Engine = Literal["sdk", "baseline"]


def platform_engine(config: ChatConfig) -> Engine:
    """The engine the platform route's turns run on: its row's.

    Only the local route runs on the fast engine, until the shared gateway
    carries it; every other route runs on the Claude Agent SDK. A disabled
    route has no engine of its own, and its SDK turn refuses itself.
    Connected accounts pick their own engine in the processor.
    """
    row = config.route_row
    return row.engine if row is not None else "sdk"
