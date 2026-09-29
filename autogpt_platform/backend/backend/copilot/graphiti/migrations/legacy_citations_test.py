"""Unit tests for reading back what an older dream episode's description
cites (``legacy_citations.py``): only the shapes the dream wrote, keys
ending it, and only uuids the graph has in the episode's own scope. The live
run is ``graphiti/backfill_derivations_integration_test.py``.
"""

import pytest

from . import backfill_derivations as backfill
from .backfill_fake import BackfillGraph, dream_row
from .legacy_citations import LegacyCitations, described_citations


class TestDescribedCitations:
    @pytest.mark.parametrize(
        "description, cited",
        [
            ("dream-pass consolidation; src_episodes=e1,e2", ([], ["e1", "e2"])),
            ("dream-pass proposal; rationale=r; src_facts=f1", (["f1"], [])),
            (
                "dream-pass proposal; src_episodes=e1; src_facts=f1,f2",
                (["f1", "f2"], ["e1"]),
            ),
            ("dream-pass consolidation; src_episodes=", ([], [])),
            ("dream-pass proposal", ([], [])),
            (None, ([], [])),
            ("dream-pass proposal; rationale=ordinary; src_facts=f9", (["f9"], [])),
        ],
        ids=[
            "consolidation",
            "proposal",
            "both kinds",
            "empty list",
            "nothing listed",
            "no description",
            "a forged key that ends it reads as written",
        ],
    )
    def test_reads_the_uuids_each_kind_lists(
        self, description: str | None, cited: tuple[list[str], list[str]]
    ) -> None:
        read_back = described_citations(description)
        assert ((read_back.facts, read_back.episodes), read_back.ambiguous) == (
            cited,
            False,
        )

    @pytest.mark.parametrize(
        "description",
        [
            "dream-pass proposal; rationale=see src_facts=x; src_facts=f1",
            "dream-pass proposal; rationale=r; src_facts=f9; src_facts=f1",
            "dream-pass proposal; rationale=a, src_facts=f9",
            "dream-pass proposal; rationale=r; src_facts=f1 and f2",
            "dream-pass consolidation; src_facts=f1; src_episodes=e1",
        ],
        ids=[
            "a key before the end",
            "a key twice",
            "a key not its own field",
            "an id with a space",
            "keys out of order",
        ],
    )
    def test_an_ambiguous_shape_attributes_nothing(self, description: str) -> None:
        assert described_citations(description) == LegacyCitations(ambiguous=True)


class TestLegacyCitationsAreChecked:
    @pytest.mark.asyncio
    async def test_a_source_the_graph_lacks_or_a_fact_of_another_scope_is_dropped(
        self,
    ) -> None:
        """The graph's sources are unscoped. A project write keeps the chat
        turn it cites (episode citations are not scoped) and loses the
        global fact."""
        project = '{"content": "x", "scope": "project:bread"}'
        driver = BackfillGraph(
            episodes=[
                dream_row("d1", "dream-pass proposal; rationale=r; src_facts=f1,ghost"),
                {
                    **dream_row("d2", "dream-pass consolidation; src_episodes=e0"),
                    "content": project,
                },
                {
                    **dream_row("d3", "dream-pass proposal; rationale=r; src_facts=f2"),
                    "content": project,
                },
                dream_row(
                    "d4",
                    "dream-pass proposal; rationale=a; b; src_facts=f1; src_facts=f2",
                ),
            ],
            facts=[],
        )

        found = await backfill.backfill_graph(driver, apply=True, cascade_forgets=False)

        [(_, records)] = driver.writes
        assert records == [
            {"uuid": "d1", "facts": ["f1"], "episodes": []},
            {"uuid": "d2", "facts": [], "episodes": ["e0"]},
            {"uuid": "d3", "facts": [], "episodes": []},
            {"uuid": "d4", "facts": [], "episodes": []},
        ]
        assert (found.rejected, found.ambiguous) == (2, 1)
