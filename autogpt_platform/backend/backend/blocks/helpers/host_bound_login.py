"""Whether a block may use a saved login for the server its inputs name.

A login with a scope is only used for the server its scope names. A login with
no scope (saved before logins had one) is only used in a graph version saved by
the user running it.
"""

from backend.data.credential_scope import ServerTarget
from backend.data.model import UserPasswordCredentials
from backend.util.clients import get_database_manager_async_client


async def login_may_go_to(
    credentials: UserPasswordCredentials,
    target: ServerTarget,
    *,
    user_id: str | None,
    graph_id: str | None,
    graph_version: int | None,
) -> bool:
    if credentials.scope is not None:
        return credentials.matches_scope(target)
    # No scope: only in a graph version the running user saved.
    if not user_id or not graph_id or graph_version is None:
        return False
    return await get_database_manager_async_client().is_graph_version_owner(
        graph_id, graph_version, user_id
    )
