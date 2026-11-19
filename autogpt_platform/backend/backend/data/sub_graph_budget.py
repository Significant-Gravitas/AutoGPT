"""How much of a workflow's sub-workflows one read may load."""

from collections.abc import Iterable, Sequence

from prisma import Prisma
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.exceptions import SubGraphsTooLargeError

MAX_SUB_GRAPHS = 200
# Nodes and links together.
MAX_SUB_GRAPH_ROWS = 10_000
# Node inputs and metadata, as stored.
MAX_SUB_GRAPH_BYTES = 10 * 1024 * 1024

# A stored graph version, with the owner a read looks it up for.
SubGraphRef = tuple[str, int, str]


class SubGraphSize(BaseModel):
    nodes: int
    links: int
    bytes: int


class SubGraphBudget(BaseModel):
    """What one read has taken so far of the sub-workflows it loads."""

    graphs: int = 0
    rows: int = 0
    bytes: int = 0

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
        f"connections, and {MAX_SUB_GRAPH_BYTES // 2**20} MB of block settings."
    )


async def get_sub_graph_sizes(
    refs: Iterable[SubGraphRef], client: Prisma | None = None
) -> dict[SubGraphRef, SubGraphSize]:
    """The size of each stored graph in *refs*, measured in the database
    without loading it. A ref to no stored graph is left out."""
    refs = list(refs)
    if not refs:
        return {}
    rows = await query_raw_with_schema(
        _SIZES_QUERY,
        [graph_id for graph_id, _, _ in refs],
        [version for _, version, _ in refs],
        [user_id for _, _, user_id in refs],
        client=client,
    )
    return {
        (row["id"], row["version"], row["userId"]): SubGraphSize(
            nodes=int(row["nodes"]), links=int(row["links"]), bytes=int(row["bytes"])
        )
        for row in rows
    }


_SIZES_QUERY = """
WITH "refs" AS (
    SELECT * FROM unnest($1::text[], $2::int[], $3::text[]) AS r("id", "version", "userId")
), "nodes" AS (
    SELECT n."id", n."agentGraphId", n."agentGraphVersion",
           octet_length(n."constantInput"::text) + octet_length(n."metadata"::text)
               AS "bytes"
      FROM {schema_prefix}"AgentNode" n
      JOIN (SELECT DISTINCT "id", "version" FROM "refs") g
        ON n."agentGraphId" = g."id" AND n."agentGraphVersion" = g."version"
)
SELECT g."id", g."version", g."userId",
       (SELECT COUNT(*) FROM "nodes" n
         WHERE n."agentGraphId" = g."id" AND n."agentGraphVersion" = g."version")
           AS "nodes",
       (SELECT COUNT(*) FROM {schema_prefix}"AgentNodeLink" l
          JOIN "nodes" n ON l."agentNodeSourceId" = n."id"
         WHERE n."agentGraphId" = g."id" AND n."agentGraphVersion" = g."version")
           AS "links",
       (SELECT COALESCE(SUM(n."bytes"), 0) FROM "nodes" n
         WHERE n."agentGraphId" = g."id" AND n."agentGraphVersion" = g."version")
           AS "bytes"
  FROM {schema_prefix}"AgentGraph" g
  JOIN "refs" r
    ON g."id" = r."id" AND g."version" = r."version" AND g."userId" = r."userId"
"""
