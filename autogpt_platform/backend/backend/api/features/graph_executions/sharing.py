"""Turning a run's public share link on and off, and deleting a run.

A run's share state, its share token and the allowlist of output files the
public link may download have to change together: the public file download
checks only the allowlist. The web app's routes and External API v2 both come
through here, so neither can update one and leave the others behind.

Every function loads the caller's own run first and raises `NotFoundError`
otherwise, so nothing is written for a run someone else owns.
"""

from datetime import datetime, timezone

from backend.data import execution as execution_db
from backend.data.sharing.tokens import generate_share_token
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

settings = Settings()


async def share_execution(user_id: str, execution_id: str) -> str:
    """Share the caller's run publicly and return its new share token.

    A previous share's file allowlist is dropped before the new token is
    written, so links from an earlier share stop working.
    """
    execution = await _own_execution(user_id, execution_id)
    share_token = generate_share_token()

    await execution_db.delete_shared_execution_files(execution_id=execution_id)
    # Owner-gated at the DB layer too: a delete racing the check above
    # surfaces as NotFoundError rather than a silent no-op.
    await execution_db.update_graph_execution_share_status(
        execution_id=execution_id,
        user_id=user_id,
        is_shared=True,
        share_token=share_token,
        shared_at=datetime.now(timezone.utc),
    )
    await execution_db.create_shared_execution_files(
        execution_id=execution_id,
        share_token=share_token,
        user_id=user_id,
        outputs=execution.outputs,
    )
    return share_token


async def unshare_execution(user_id: str, execution_id: str) -> None:
    """Revoke the run's public link and every file download it allowed."""
    await _own_execution(user_id, execution_id)

    await execution_db.delete_shared_execution_files(execution_id=execution_id)
    await execution_db.update_graph_execution_share_status(
        execution_id=execution_id,
        user_id=user_id,
        is_shared=False,
        share_token=None,
        shared_at=None,
    )


async def delete_execution(user_id: str, execution_id: str) -> None:
    """Delete the caller's run, revoking the file downloads a share allowed.

    The shared page already hides deleted runs, but the file download checks
    only the allowlist, which would otherwise outlive the run.
    """
    await execution_db.delete_graph_execution(
        graph_exec_id=execution_id, user_id=user_id
    )
    await execution_db.delete_shared_execution_files(execution_id=execution_id)


def share_url(share_token: str) -> str:
    frontend_url = settings.config.frontend_base_url or "http://localhost:3000"
    return f"{frontend_url}/share/{share_token}"


async def _own_execution(
    user_id: str, execution_id: str
) -> execution_db.GraphExecution:
    execution = await execution_db.get_graph_execution(
        user_id=user_id, execution_id=execution_id
    )
    if execution is None:
        raise NotFoundError(f"Run #{execution_id} not found")
    return execution
