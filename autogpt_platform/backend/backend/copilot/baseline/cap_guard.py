"""A delegated thread's hand-off cap, kept by the baseline engine mid-turn.

The SDK engine lands a turn on its budget through the turn governor, and a
delegated thread's budget is cut to what its cap has left. The baseline
engine has no governor, so it keeps the cap itself: between tool rounds it
compares what the turn has spent with what the cap had left when the turn
started, and once that is gone it runs no further round and ends the turn as
a pause at the cap. The executor then holds the thread for the user (see
``delegation_cap.hold_at_cap``), as it does for the SDK engine.

Between rounds, not inside one: a round already under way finishes, so a
turn can pass its cap by at most the round that crossed it.
"""

from pydantic import BaseModel

from backend.copilot.model import ChatSession
from backend.copilot.turn_budget import delegation_cap_left_usd
from backend.copilot.turn_pause import TurnPause


class CapGuard(BaseModel):
    cap_usd: float
    # What the cap had left when the turn started.
    left_usd: float

    def reached(self, turn_spent_usd: float | None) -> bool:
        return (turn_spent_usd or 0.0) >= self.left_usd

    def pause(self) -> TurnPause:
        # The engine stopped the turn between rounds; the model wrote no
        # checkpoint for it, so the card must not claim one.
        return TurnPause(reason="task", checkpoint_saved=False, cap_usd=self.cap_usd)


async def cap_guard_for(user_id: str | None, session: ChatSession) -> CapGuard | None:
    """The guard for a delegated thread's turn; None when it has no cap."""
    cap = session.metadata.delegation_cap_usd
    if cap is None or not user_id:
        return None
    left = await delegation_cap_left_usd(user_id, session, cap)
    return CapGuard(cap_usd=cap, left_usd=left)
