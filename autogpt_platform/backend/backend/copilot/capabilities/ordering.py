"""Service-aware ordering of capability search hits."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .ranking import tier

if TYPE_CHECKING:
    from .index import SearchHit

CONNECTED_COVERAGE_MARGIN = 1.0


def rank_hits(
    hits: list[SearchHit], *, lift: bool = False, mcp_first: bool = False
) -> list[SearchHit]:
    """Rank coverage before connection, first-party tools, and weighted BM25.

    ``lift`` applies only after filtering to a named service. It prioritizes
    tools on coverage ties and lets a connected MCP server cover one fewer
    concept: server descriptions name the service, not every action it offers.
    Queries without a service retain strict coverage and connection ordering.

    ``mcp_first`` replaces that order for expert service searches: connection
    leads, with MCP ahead of the same service's blocks within a tier.
    """
    if mcp_first:
        return _mcp_order(hits)
    return _coverage_order(hits, lift=lift)


def _mcp_order(hits: list[SearchHit]) -> list[SearchHit]:
    ranked = sorted(
        hits,
        key=lambda h: (
            tier(h.entry, h.connected),
            -h.coverage,
            h.entry.kind != "tool",
            -h.score,
            h.entry.name.lower(),
        ),
    )
    service_hits: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in ranked:
        if hit.entry.service:
            service_hits[hit.entry.service].append(hit)
    for service, entries in service_hits.items():
        service_hits[service] = sorted(entries, key=_mcp_service_key)
    offsets: dict[str, int] = defaultdict(int)
    result: list[SearchHit] = []
    for hit in ranked:
        service = hit.entry.service
        if not service:
            result.append(hit)
            continue
        result.append(service_hits[service][offsets[service]])
        offsets[service] += 1
    return result


def _mcp_service_key(hit: SearchHit) -> tuple[int, bool, float, bool, float, str]:
    return (
        tier(hit.entry, hit.connected),
        hit.entry.kind != "mcp_server",
        -hit.coverage,
        hit.entry.kind != "tool",
        -hit.score,
        hit.entry.name.lower(),
    )


def _coverage_order(hits: list[SearchHit], *, lift: bool) -> list[SearchHit]:
    best = max((h.coverage for h in hits), default=0.0)

    def key(hit: SearchHit) -> tuple[float, int, int, float, str]:
        rank = tier(hit.entry, hit.connected)
        not_tool = int(hit.entry.kind != "tool")
        coverage = hit.coverage
        if (
            lift
            and hit.entry.kind == "mcp_server"
            and rank == 0
            and coverage >= best - CONNECTED_COVERAGE_MARGIN
        ):
            coverage = best
        first, second = (not_tool, rank) if lift else (rank, not_tool)
        return (-coverage, first, second, -hit.score, hit.entry.name.lower())

    return sorted(hits, key=key)
