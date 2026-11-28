"""How much of a workflow's sub-workflows one read may load."""

from collections.abc import Iterable, Sequence

from prisma import Prisma
from prisma.errors import RawQueryError
from pydantic import BaseModel

from backend.data.db import bounded_transaction, query_raw_with_schema
from backend.util.exceptions import SubGraphsTooLargeError

MAX_SUB_GRAPHS = 200
# Nodes and links together.
MAX_SUB_GRAPH_ROWS = 10_000
# Node inputs and metadata, and link pin names, as stored.
MAX_SUB_GRAPH_BYTES = 10 * 1024 * 1024

# A stored graph version, with the owner a read looks it up for.
SubGraphRef = tuple[str, int, str]
# How long the database may spend sizing one level before the read is refused.
SIZE_QUERY_TIMEOUT_MS = 5_000


class SubGraphSize(BaseModel):
    nodes: int
    links: int
    bytes: int


class SubGraphBudget(BaseModel):
    """What one read has taken so far of the sub-workflows it loads."""

    graphs: int = 0
    rows: int = 0
    bytes: int = 0

    def can_name(self, count: int) -> bool:
        """Whether *count* more sub-workflows could fit, checked before they
        are sized."""
        return self.graphs + count <= MAX_SUB_GRAPHS

    def take(self, sizes: Sequence[SubGraphSize]) -> bool:
        """Count *sizes* in if they fit, and say whether they did."""
        graphs = self.graphs + len(sizes)
        rows = self.rows + sum(size.nodes + size.links for size in sizes)
        nbytes = self.bytes + sum(size.bytes for size in sizes)
        if (
            graphs > MAX_SUB_GRAPHS
            or rows > MAX_SUB_GRAPH_ROWS
            or nbytes > MAX_SUB_GRAPH_BYTES
        ):
            return False
        self.graphs, self.rows, self.bytes = graphs, rows, nbytes
        return True


def too_large_error() -> SubGraphsTooLargeError:
    return SubGraphsTooLargeError(
        "This workflow's sub-workflows are too large to load together: at most "
        f"{MAX_SUB_GRAPHS} sub-workflows, {MAX_SUB_GRAPH_ROWS:,} blocks and "
        f"connections, and {MAX_SUB_GRAPH_BYTES // 2**20} MB of block settings and "
        "connection names."
    )


async def get_sub_graph_sizes(
    refs: Iterable[SubGraphRef], client: Prisma | None = None
) -> dict[SubGraphRef, SubGraphSize]:
    """The size of each stored graph in *refs*, measured in the database
    without loading it. A ref to no stored graph is left out. Raises
    `SubGraphsTooLargeError` if the measuring takes too long."""
    refs = list(refs)
    if not refs:
        return {}
    args = (
        [graph_id for graph_id, _, _ in refs],
        [version for _, version, _ in refs],
        [user_id for _, _, user_id in refs],
    )
    try:
        if client is None:
            async with bounded_transaction(SIZE_QUERY_TIMEOUT_MS) as tx:
                rows = await query_raw_with_schema(_SIZES_QUERY, *args, client=tx)
        else:
            # Inside a caller's transaction, which carries its own timeout.
            rows = await query_raw_with_schema(_SIZES_QUERY, *args, client=client)
    except RawQueryError as e:
        if "statement timeout" not in str(e):
            raise
        raise too_large_error() from e
    return {
        (row["id"], row["version"], row["userId"]): SubGraphSize(
            nodes=int(row["nodes"]), links=int(row["links"]), bytes=int(row["bytes"])
        )
        for row in rows
    }


_SIZES_QUERY = """
WITH "graphs" AS (
    SELECT g."id", g."version", g."userId"
      FROM {schema_prefix}"AgentGraph" g
      JOIN unnest($1::text[], $2::int[], $3::text[]) AS r("id", "version", "userId")
        ON g."id" = r."id" AND g."version" = r."version" AND g."userId" = r."userId"
), "nodes" AS (
    SELECT n."id", n."agentGraphId", n."agentGraphVersion",
           octet_length(n."constantInput"::text) + octet_length(n."metadata"::text)
               AS "bytes"
      FROM {schema_prefix}"AgentNode" n
      JOIN "graphs" g
        ON n."agentGraphId" = g."id" AND n."agentGraphVersion" = g."version"
), "node_totals" AS (
    SELECT "agentGraphId", "agentGraphVersion", COUNT(*) AS "nodes",
           SUM("bytes") AS "bytes"
      FROM "nodes"
     GROUP BY "agentGraphId", "agentGraphVersion"
), "link_totals" AS (
    -- A read loads each link with both of the nodes it joins.
    SELECT n."agentGraphId", n."agentGraphVersion", COUNT(*) AS "links",
           SUM(2 * (octet_length(l."sourceName") + octet_length(l."sinkName")))
               AS "bytes"
      FROM {schema_prefix}"AgentNodeLink" l
      JOIN "nodes" n ON l."agentNodeSourceId" = n."id"
     GROUP BY n."agentGraphId", n."agentGraphVersion"
)
SELECT g."id", g."version", g."userId",
       COALESCE(nt."nodes", 0) AS "nodes",
       COALESCE(lt."links", 0) AS "links",
       COALESCE(nt."bytes", 0) + COALESCE(lt."bytes", 0) AS "bytes"
  FROM "graphs" g
  LEFT JOIN "node_totals" nt
    ON nt."agentGraphId" = g."id" AND nt."agentGraphVersion" = g."version"
  LEFT JOIN "link_totals" lt
    ON lt."agentGraphId" = g."id" AND lt."agentGraphVersion" = g."version"
"""
