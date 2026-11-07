"""Wall-clock room for Exa blocks that wait longer than the executor allows by default.

The executor cancels a block's ``run`` after ``execution_timeout_seconds``
(30 minutes unless the block overrides it). A block whose own timeout input
goes past that would be cancelled before its graceful timeout output fires, so
each such block sets ``execution_timeout_seconds`` from its input's ceiling,
as the Conductor and timer blocks do.
"""

from backend.blocks._base import DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS

# Longest wait an Exa Agent run gets: ultra runs can take up to 3 hours.
MAX_AGENT_WAIT_SECONDS = 3 * 60 * 60
# Longest wait the legacy Research blocks offer.
MAX_RESEARCH_WAIT_SECONDS = 60 * 60
# Longest wait the Websets wait blocks offer.
MAX_WEBSETS_WAIT_SECONDS = 30 * 60
# The requests made around the wait: the create call, the final fetch after
# the deadline, failure events, and their bounded retries.
WAIT_SLACK_SECONDS = 5 * 60


def execution_timeout_for(max_wait_seconds: int) -> int:
    """An executor cap that lets a wait of ``max_wait_seconds`` finish."""
    return max(
        DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS, max_wait_seconds + WAIT_SLACK_SECONDS
    )
