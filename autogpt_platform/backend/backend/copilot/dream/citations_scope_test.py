"""A dream write keeps only the citations whose source shares its scope
(``citations.py``): the rule the prompts give the model ("group by scope",
"proposed findings live in the same scope as their evidence"), enforced by
apply. The live run is ``graphiti/recall_citation_scope_integration_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.memory_model import MemoryEnvelope
from backend.copilot.graphiti.recall_citations import Citations
from backend.copilot.graphiti.scope import MemoryScope

from . import apply as apply_mod
from .citations import UNSCOPED, scope_key, source_scopes, validated_citations
from .fetch import DreamInput, EpisodeRow, FactRow
from .schemas import ConsolidatedFact, DreamOperations, ProposedFinding

_SCOPE = MemoryScope.for_user("u-1234567890ab")
_SCOPES = {
    "f-global": UNSCOPED,
    "f-bread": "project:bread",
    "ep-bread": "project:bread",
}
_READ = {
    "known_fact_uuids": {"f-global", "f-bread"},
    "known_episode_uuids": {"ep-bread"},
    "source_scopes": _SCOPES,
}


def _check(scope: str, *cited: str):
    return validated_citations(
        [uuid for uuid in cited if uuid.startswith("f")],
        [uuid for uuid in cited if uuid.startswith("ep")],
        scope=scope,
        known_facts={"f-global", "f-bread"},
        known_episodes={"ep-bread"},
        source_scopes=_SCOPES,
    )


class TestValidatedCitations:
    def test_a_citation_of_another_scope_is_dropped_and_counted(self) -> None:
        checked = _check("project:bread", "f-bread", "f-global", "ep-bread")

        assert checked.citations == Citations(
            fact_uuids=["f-bread"], episode_uuids=["ep-bread"]
        )
        assert checked.cross_scope == 1

    def test_a_write_citing_only_another_scope_cites_nothing(self) -> None:
        """Codex's probe: a project conclusion citing a global fact."""
        checked = _check("project:unrelated", "f-global")

        assert (checked.citations, checked.cross_scope) == (None, 1)

    def test_an_unknown_uuid_is_not_counted_as_cross_scope(self) -> None:
        checked = _check("project:bread", "made-up")

        assert (checked.citations, checked.cross_scope) == (None, 0)

    def test_a_source_with_no_scope_given_is_unscoped(self) -> None:
        checked = validated_citations(
            ["f1"],
            [],
            scope="real:global",
            known_facts={"f1"},
            known_episodes=set(),
            source_scopes={},
        )

        assert checked.citations == Citations(fact_uuids=["f1"])

    def test_scopes_compare_as_the_prompts_list_them(self) -> None:
        assert scope_key(" project:bread\n") == "project:bread"
        assert scope_key(None) == scope_key("") == UNSCOPED


def test_a_pass_reads_each_sources_scope() -> None:
    """A fact's own (unset: unscoped), an episode's envelope's, and a chat
    turn, plain text, unscoped."""
    envelope = MemoryEnvelope(content="bread notes", scope="project:bread")
    bundle = DreamInput(
        user_id="u",
        group_id="g",
        window_start=datetime(2026, 9, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 9, 28, tzinfo=timezone.utc),
        facts=[_fact("f-bread", "project:bread"), _fact("f-old", None)],
        episodes=[
            _episode("ep-stored", envelope.model_dump_json()),
            _episode("ep-chat", "Nick: I bake on Fridays"),
            _episode("ep-empty", None),
        ],
    )

    assert source_scopes(bundle) == {
        "f-bread": "project:bread",
        "f-old": UNSCOPED,
        "ep-stored": "project:bread",
        "ep-chat": UNSCOPED,
        "ep-empty": UNSCOPED,
    }


@pytest.fixture
def enqueue(mocker) -> AsyncMock:
    queued = AsyncMock(return_value=True)
    mocker.patch.object(apply_mod, "enqueue_episode", queued)
    mocker.patch.object(apply_mod, "wait_for_ingestion", AsyncMock(return_value=True))
    mocker.patch.object(apply_mod, "_create_dream_session", AsyncMock(return_value="s"))
    mocker.patch.object(apply_mod, "_write_dream_summary_message", AsyncMock())
    return queued


@pytest.mark.asyncio
async def test_apply_drops_the_citations_of_another_scope(enqueue: AsyncMock) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="Sunrise Bakery changes suppliers every quarter",
                scope="project:unrelated",
                confidence=0.9,
                source_fact_uuids=["f-global"],
            ),
            ConsolidatedFact(
                content="The bread project uses the new flour",
                scope="project:bread",
                confidence=0.9,
                source_fact_uuids=["f-bread", "f-global"],
            ),
        ],
        proposals=[
            ProposedFinding(
                content="The bread project may need a second mill",
                scope="project:bread",
                confidence=0.5,
                rationale="volume",
                source_episode_uuids=["ep-bread"],
            )
        ],
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    assert [call.kwargs["citations"] for call in enqueue.await_args_list] == [
        Citations(fact_uuids=["f-bread"]),
        Citations(episode_uuids=["ep-bread"]),
    ]
    assert (
        stats["uncited_writes_dropped"],
        stats["cross_scope_citations_dropped"],
    ) == (
        1,
        2,
    )


@pytest.mark.asyncio
async def test_an_empty_pass_still_counts_what_it_dropped(enqueue: AsyncMock) -> None:
    ops = DreamOperations(
        writes=[
            ConsolidatedFact(
                content="a project conclusion from a global fact",
                scope="project:unrelated",
                confidence=0.9,
                source_fact_uuids=["f-global"],
            )
        ]
    )

    stats = await apply_mod.apply_operations(_SCOPE, "p1", ops, **_READ)

    enqueue.assert_not_awaited()
    assert (
        stats["uncited_writes_dropped"],
        stats["cross_scope_citations_dropped"],
    ) == (
        1,
        1,
    )


def _fact(uuid: str, scope: str | None) -> FactRow:
    return FactRow(
        uuid=uuid,
        source="Nick",
        target="Bread",
        name="bakes",
        fact="Nick bakes bread",
        scope=scope,
        confidence=0.8,
        status="active",
        created_at=None,
    )


def _episode(uuid: str, content: str | None) -> EpisodeRow:
    return EpisodeRow(
        uuid=uuid,
        name="n",
        content=content,
        source_description=None,
        valid_at=None,
        created_at=None,
    )
