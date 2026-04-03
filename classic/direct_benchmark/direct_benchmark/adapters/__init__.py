"""Benchmark adapter registry and factory."""

from typing import Optional

# Registry of benchmark adapters
_ADAPTER_REGISTRY: dict[str, type["BenchmarkAdapter"]] = {}  # noqa: F821


def register_adapter(name: str):
    """Decorator to register a benchmark adapter.

    Usage:
        @register_adapter("gaia")
        class GAIAAdapter(BenchmarkAdapter):
            ...
    """

    def decorator(cls: type["BenchmarkAdapter"]) -> type["BenchmarkAdapter"]:
        _ADAPTER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_adapter(name: str) -> Optional[type["BenchmarkAdapter"]]:
    """Get an adapter class by name.

    Args:
        name: The benchmark name (e.g., "gaia", "swe-bench", "agent-bench").

    Returns:
        The adapter class, or None if not found.
    """
    return _ADAPTER_REGISTRY.get(name.lower())


def list_adapters() -> list[str]:
    """List all registered adapter names."""
    return list(_ADAPTER_REGISTRY.keys())


# Import adapters to trigger registration
# These imports are at the bottom to avoid circular imports
from .agent_bench import AgentBenchAdapter  # noqa: E402, F401
from .base import BenchmarkAdapter  # noqa: E402, F401
from .gaia import GAIAAdapter  # noqa: E402, F401
from .swe_bench import SWEBenchAdapter  # noqa: E402, F401

__all__ = [
    "BenchmarkAdapter",
    "GAIAAdapter",
    "SWEBenchAdapter",
    "AgentBenchAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
