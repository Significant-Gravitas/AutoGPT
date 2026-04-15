"""Tests for graphiti_search helper functions."""

from types import SimpleNamespace

from backend.copilot.graphiti.memory_model import MemoryEnvelope, MemoryKind, SourceKind
from backend.copilot.tools.graphiti_search import (
    _filter_episodes_by_scope,
    _format_episodes,
)


class TestFilterEpisodesByScopeTruncation:
    """extract_episode_body() truncates to 500 chars.  A MemoryEnvelope
    with a long content field exceeds that limit, producing invalid JSON.
    _filter_episodes_by_scope then treats it as a plain-text episode
    (real:global), leaking project-scoped data into global results.
    """

    def test_long_envelope_filtered_by_scope(self) -> None:
        envelope = MemoryEnvelope(
            content="x" * 600,
            source_kind=SourceKind.user_asserted,
            scope="project:crm",
            memory_kind=MemoryKind.fact,
        )
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        # Requesting real:global scope — this project:crm episode should be excluded
        results = _filter_episodes_by_scope([ep], "real:global")
        assert (
            results == []
        ), f"project-scoped episode leaked into global results: {results}"

    def test_short_envelope_filtered_correctly(self) -> None:
        """Short envelopes (under 500 chars) are parsed correctly."""
        envelope = MemoryEnvelope(
            content="short note",
            scope="project:crm",
        )
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        results = _filter_episodes_by_scope([ep], "real:global")
        assert results == []


class TestRedundantFormatting:
    """_format_episodes is called even when scope filter will overwrite it.
    Not a correctness bug, but verify the scope path doesn't depend on it.
    """

    def test_scope_filter_independent_of_format_episodes(self) -> None:
        envelope = MemoryEnvelope(content="note", scope="real:global")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        from_format = _format_episodes([ep])
        from_scope = _filter_episodes_by_scope([ep], "real:global")
        assert len(from_format) == 1
        assert len(from_scope) == 1
