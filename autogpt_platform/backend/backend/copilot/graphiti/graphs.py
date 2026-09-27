"""Enumerate the memory graphs on the FalkorDB instance.

Nothing records which memory scopes exist — drivers never create a graph
until the first write — so ``GRAPH.LIST`` is the only inventory of accounts
that have memory. Expert graphs are named by a digest of the expert id and
cannot be mapped back; list experts from Postgres instead.
"""

from falkordb.asyncio import FalkorDB

from .client import derive_group_id
from .config import graphiti_config

ACCOUNT_GRAPH_PREFIX = "user_"


async def list_graph_names() -> list[str]:
    """Every graph on the FalkorDB instance, via ``GRAPH.LIST``."""
    client = FalkorDB(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
    )
    try:
        return list(await client.list_graphs())
    finally:
        await client.aclose()


async def list_account_graph_owners() -> list[str]:
    """The user ids that own an account graph (``user_<id>``), sorted."""
    names = await list_graph_names()
    return sorted({owner for name in names if (owner := account_graph_owner(name))})


def account_graph_owner(graph_name: str) -> str | None:
    """The user id whose account graph ``graph_name`` is, or None for any
    other graph (an expert's, or one this code did not derive)."""
    if not graph_name.startswith(ACCOUNT_GRAPH_PREFIX):
        return None
    user_id = graph_name.removeprefix(ACCOUNT_GRAPH_PREFIX)
    try:
        return user_id if derive_group_id(user_id) == graph_name else None
    except ValueError:
        return None
