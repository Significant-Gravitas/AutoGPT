"""Keep one-time sandbox notices out of the CLI's resumed history."""

from backend.copilot.cli_session_entry import rewrite_user_entry
from backend.copilot.sandbox_status import (
    SANDBOX_REPLACED_NOTICE,
    SANDBOX_REPLACED_WITHOUT_WORKSPACE_NOTICE,
    SANDBOX_STATUS_TAG,
)


def rewrite_sandbox_status_entry(entry: object, line: str) -> dict[str, object] | None:
    if SANDBOX_STATUS_TAG not in line:
        return None
    return rewrite_user_entry(entry, _without_notices)


def _without_notices(text: str) -> str:
    return text.replace(SANDBOX_REPLACED_NOTICE, "").replace(
        SANDBOX_REPLACED_WITHOUT_WORKSPACE_NOTICE, ""
    )
