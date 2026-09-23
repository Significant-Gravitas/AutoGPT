"""Unified capability registry for the copilot: tools, blocks, MCP servers
and the session owner's skills behind one index, searched by
``find_capability`` and executed by ``run_capability``."""

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
