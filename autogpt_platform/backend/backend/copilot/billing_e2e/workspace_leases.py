"""The live workspace leases a turn and its tools hold, granted.

The executor holds a lease on the session's workspace while a turn runs, and
each tool call takes its own (``live_resource_lease``), both from the database
manager (``credit_db``). The suite runs without one and its users may run in
their own workspace, so every ``credit_db`` it stands in grants them.
"""

from unittest.mock import AsyncMock


def granted_leases() -> dict[str, AsyncMock]:
    return {
        "acquire_live_resource_lease": AsyncMock(return_value="lease-1"),
        "release_live_resource_lease": AsyncMock(return_value=True),
        "is_live_resource_lease_active": AsyncMock(return_value=True),
    }
