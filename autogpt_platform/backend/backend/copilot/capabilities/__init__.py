"""Unified capability registry for the copilot: tools, blocks and MCP servers
behind one index, searched by ``find_capability`` and executed by
``run_capability`` (both added in the follow-up change)."""

from .index import CapabilityIndex, SearchHit, SearchResult
from .models import CapabilityEntry, Connection, Implementation
from .ranking import ConnectionState
from .registry import build_entries, get_registry, reset_registry

__all__ = [
    "CapabilityEntry",
    "CapabilityIndex",
    "Connection",
    "ConnectionState",
    "Implementation",
    "SearchHit",
    "SearchResult",
    "build_entries",
    "get_registry",
    "reset_registry",
]
