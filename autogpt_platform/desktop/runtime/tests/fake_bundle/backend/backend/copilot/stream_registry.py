"""What the host calls to close the chat of an AutoPilot turn that was ended
without its executor: upstream's signature, and a marker file per call."""

import os
from pathlib import Path

CANCELLED_MESSAGE = "Operation cancelled"


async def mark_session_completed(
    session_id: str,
    error_message: str | None = None,
    *,
    skip_error_publish: bool = False,
    turn_id: str = "",
) -> bool:
    marks = Path(os.environ["FAKE_MARKS"])
    refusals = marks / "closing-refused"
    if refusals.exists():
        # As when database-manager does not answer yet: the first calls fail.
        left = int(refusals.read_text())
        if left > 0:
            refusals.write_text(str(left - 1))
            raise ConnectionError("database-manager does not answer yet")
    (marks / f"closed-{session_id}").write_text(
        f"{error_message}|{skip_error_publish}|{turn_id}"
    )
    return True
