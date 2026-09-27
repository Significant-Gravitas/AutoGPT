"""Unit tests for how recalled memory is written out (``recall_render.py``)."""

from datetime import datetime, timezone

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from . import recall_render
from .memory_model import MemoryEnvelope, MemoryKind, SourceKind
from .scope import MemoryScope

_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_SCOPE = MemoryScope.for_user("user-abc")


def _fact(
    *,
    status: str = "active",
    expired_at: datetime | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    fact: str = "Alice works on Atlas",
) -> EntityEdge:
    return EntityEdge(
        group_id=_SCOPE.group_id,
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=_NOW,
        name="works_on",
        fact=fact,
        expired_at=expired_at,
        valid_at=valid_at,
        invalid_at=invalid_at,
        attributes={"status": status},
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep1",
        group_id=_SCOPE.group_id,
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


class TestRender:
    def test_live_fact_shows_its_validity(self) -> None:
        fact = _fact(valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert recall_render.render(fact) == (
            "Alice works on Atlas (valid: 2025-01-01 00:00:00+00:00 — present)"
        )

    def test_live_fact_whose_valid_time_ended_reads_as_history(self) -> None:
        """Not expired, so recall still returns it: it held until
        ``invalid_at``. The interval shows that end instead of "present",
        and no retirement label is given, because it was never retired."""
        fact = _fact(
            valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            invalid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert recall_render.render(fact) == (
            "Alice works on Atlas "
            "(valid: 2024-01-01 00:00:00+00:00 — 2025-01-01 00:00:00+00:00)"
        )

    @pytest.mark.parametrize("status", ["retracted", "superseded", "contradicted"])
    def test_retired_fact_is_labelled_never_present(self, status: str) -> None:
        fact = _fact(status=status, expired_at=_NOW)
        rendered = recall_render.render(fact)
        assert rendered == f"Alice works on Atlas ({status} 2026-09-26 12:00:00+00:00)"
        assert "present" not in rendered

    def test_expired_fact_with_a_live_status_is_labelled_expired(self) -> None:
        # What graphiti's own contradiction handling leaves behind: expired,
        # status never changed.
        fact = _fact(status="active", expired_at=_NOW)
        assert recall_render.render(fact) == (
            "Alice works on Atlas (expired 2026-09-26 12:00:00+00:00)"
        )

    def test_retired_status_without_a_timestamp(self) -> None:
        fact = _fact(status="retracted")
        assert recall_render.render(fact) == (
            "Alice works on Atlas (retracted at an unknown time)"
        )

    def test_fact_text_falls_back_to_the_relation_name(self) -> None:
        assert recall_render.fact_text(_fact(fact="")) == "works_on"


class TestRenderEpisode:
    def test_timestamp_and_body(self) -> None:
        assert recall_render.render_episode(_episode("talked about coffee")) == (
            "[2026-09-26 12:00:00+00:00] talked about coffee"
        )

    def test_body_cut_to_display_length(self) -> None:
        body = "x" * recall_render.EPISODE_DISPLAY_CHARS
        assert recall_render.render_episode(_episode("x" * 1000)) == (
            f"[2026-09-26 12:00:00+00:00] {body}"
        )


class TestEpisodeScope:
    @pytest.mark.parametrize(
        "content",
        ["plain conversation text", "[1, 2, 3]", '"just a string"', "null"],
    )
    def test_non_envelope_bodies_are_global(self, content: str) -> None:
        scope = recall_render.episode_scope(_episode(content))
        assert scope == recall_render.GLOBAL_SCOPE

    def test_envelope_scope_is_read(self) -> None:
        envelope = MemoryEnvelope(content="project note", scope="project:crm")
        episode = _episode(envelope.model_dump_json())
        assert recall_render.episode_scope(episode) == "project:crm"

    def test_envelope_without_scope_is_global(self) -> None:
        episode = _episode('{"content": "x"}')
        assert recall_render.episode_scope(episode) == "real:global"

    def test_long_envelope_is_parsed_whole(self) -> None:
        """An envelope longer than the display cut is still parsed: scoping
        the truncated body would leak a project memory into global recall."""
        envelope = MemoryEnvelope(
            content="x" * 600,
            source_kind=SourceKind.user_asserted,
            scope="project:crm",
            memory_kind=MemoryKind.fact,
        )
        body = envelope.model_dump_json()
        assert len(body) > recall_render.EPISODE_DISPLAY_CHARS

        assert recall_render.episode_scope(_episode(body)) == "project:crm"
