"""Unit tests for ``recall_ingest``: graphiti's edge dedup never names a
forgotten edge, checked against graphiti's own dedup prompt, and a prompt the
guard cannot read in full names no edge at all.

The live runs, through the production worker, are
``recall_ingest_integration_test.py`` and ``recall_reteach_integration_test.py``.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.token_tracker import TokenUsageTracker
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.models import Message
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

from . import recall_ingest
from .client import _build_graphiti
from .recall import FORGOTTEN_FACT
from .recall_ingest import ForgetAwareLLMClient, forgotten_candidates, off_forgotten

_EXISTING = "EXISTING FACTS"
_CANDIDATES = "FACT INVALIDATION CANDIDATES"
# A forgotten and a live existing fact, then one invalidation candidate; the
# model names the live ones, which only an unreadable prompt takes away.
_NAMES_LIVE_EDGES = {"duplicate_facts": [1], "contradicted_facts": [2]}
_NAMES_NOTHING = {"duplicate_facts": [], "contradicted_facts": []}


@pytest.fixture(autouse=True)
def reported_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no prompt shape logged yet."""
    monkeypatch.setattr(recall_ingest, "_REPORTED_SHAPES", set())


def _dedup_prompt(existing: list[str], candidates: list[str]) -> list[Message]:
    """graphiti's edge dedup prompt, one idx numbering across both lists."""
    return prompt_library.dedupe_edges.resolve_edge(
        {
            "existing_edges": [{"idx": i, "fact": f} for i, f in enumerate(existing)],
            "new_edge": "Alice is assigned to work on the Atlas project",
            "edge_invalidation_candidates": [
                {"idx": len(existing) + i, "fact": f} for i, f in enumerate(candidates)
            ],
        }
    )


def _mixed_prompt() -> list[Message]:
    return _dedup_prompt([FORGOTTEN_FACT, "Alice owns the Atlas budget"], ["Bob"])


def _edited(prompt: list[Message], old: str, new: str) -> list[Message]:
    return [
        m.model_copy(update={"content": m.content.replace(old, new)}) for m in prompt
    ]


def _retagged(prompt: list[Message], tag: str, new: str) -> list[Message]:
    """``prompt`` with the ``tag`` pair printed as ``new`` (gone when empty)."""
    opened = _edited(prompt, f"<{tag}>", f"<{new}>" if new else "")
    return _edited(opened, f"</{tag}>", f"</{new}>" if new else "")


def _inner(answer: dict[str, Any]) -> MagicMock:
    inner = MagicMock(spec=LLMClient)
    inner.config = LLMConfig()
    inner.token_tracker = TokenUsageTracker()
    inner.generate_response = AsyncMock(return_value=answer)
    return inner


class TestForgottenCandidates:
    def test_finds_every_forgotten_edge_in_both_lists(self) -> None:
        prompt = _dedup_prompt(
            [FORGOTTEN_FACT, "Alice owns the Atlas budget"],
            ["Bob leads Atlas", FORGOTTEN_FACT],
        )

        assert forgotten_candidates(prompt) == {0, 3}

    def test_a_prompt_without_one_names_none(self) -> None:
        assert (
            forgotten_candidates(_dedup_prompt(["Alice works on Atlas"], [])) == set()
        )

    def test_quotes_in_other_facts_do_not_confuse_it(self) -> None:
        prompt = _dedup_prompt(['Alice\'s "big" budget', FORGOTTEN_FACT], [])

        assert forgotten_candidates(prompt) == {1}

    def test_lists_it_cannot_read_are_reported_as_such(self) -> None:
        prompt = [Message(role="user", content=f"<EXISTING FACTS>\n{FORGOTTEN_FACT}")]

        assert forgotten_candidates(prompt) is None

    def test_a_forgotten_candidate_outside_the_two_lists_is_still_found(
        self,
    ) -> None:
        """By its text, not its tags: a list the guard does not know about."""
        extra = Message(
            role="user",
            content=f"<RELATED FACTS>\n[{{'idx': 2, 'fact': 'Carol owns Borealis', "
            f"'name': '{FORGOTTEN_FACT}'}}, {{'idx': 3, 'fact': '{FORGOTTEN_FACT}'}}]"
            "\n</RELATED FACTS>",
        )
        prompt = [*_dedup_prompt([FORGOTTEN_FACT], ["Bob leads Atlas"]), extra]

        assert forgotten_candidates(prompt) == {0, 2, 3}

    def test_with_the_tags_gone_the_placeholder_still_marks_its_candidate(
        self,
    ) -> None:
        prompt = _retagged(_retagged(_mixed_prompt(), _EXISTING, ""), _CANDIDATES, "")
        text = "\n".join(message.content for message in prompt)

        assert recall_ingest._shown(text) == {0}
        assert forgotten_candidates(prompt) is None

    def test_a_placeholder_outside_any_candidate_names_no_edge(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both lists read, and it is logged at error like any prompt the
        guard cannot read in full, once for the shape
        (``r6-dedup-shapes.py``, ``placeholder_outside_candidate``)."""
        prompt = _edited(
            _mixed_prompt(), "NEW FACT>\n", f"NEW FACT>\n{FORGOTTEN_FACT} "
        )

        assert forgotten_candidates(prompt) is None
        assert forgotten_candidates(prompt) is None
        [error] = [r for r in caplog.records if r.levelname == "ERROR"]
        assert "outside a candidate" in error.getMessage()


class TestAnUnreadablePrompt:
    """Codex's r5 ``corrupt_partial`` and its relatives: when the prompt
    shows the placeholder, anything short of both lists, once each and
    readable, names no edge, so the statement is saved as new."""

    @pytest.mark.parametrize(
        "damage",
        [
            lambda p: _retagged(p, _EXISTING, "EXISTING_EDGES"),
            lambda p: _retagged(p, _CANDIDATES, "INVALIDATION_EDGES"),
            lambda p: [
                *p,
                Message(role="user", content=f"<{_EXISTING}>[]</{_EXISTING}>"),
            ],
            lambda p: _edited(p, f"<{_CANDIDATES}>\n[", f"<{_CANDIDATES}>\n[oops, "),
            lambda p: _edited(p, "'fact': 'Bob'", "'text': 'Bob'"),
        ],
        ids=[
            "existing renamed",
            "candidates renamed",
            "shown twice",
            "unparsable",
            "no fact",
        ],
    )
    @pytest.mark.asyncio
    async def test_names_no_edge(
        self, damage: Callable[[list[Message]], list[Message]]
    ) -> None:
        client = ForgetAwareLLMClient(_inner(dict(_NAMES_LIVE_EDGES)))

        answer = await client.generate_response(damage(_mixed_prompt()), EdgeDuplicate)

        assert answer == _NAMES_NOTHING

    @pytest.mark.asyncio
    async def test_the_intact_prompt_keeps_the_live_edges_named(self) -> None:
        client = ForgetAwareLLMClient(_inner(dict(_NAMES_LIVE_EDGES)))

        answer = await client.generate_response(_mixed_prompt(), EdgeDuplicate)

        assert answer == _NAMES_LIVE_EDGES

    def test_is_logged_at_error_once_per_shape(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        renamed = _retagged(_mixed_prompt(), _EXISTING, "EXISTING_EDGES")
        for prompt in (renamed, renamed, _retagged(_mixed_prompt(), _CANDIDATES, "")):
            forgotten_candidates(prompt)

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 2
        assert "EXISTING_EDGES" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_without_the_placeholder_passes_the_answer_but_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        prompt = _retagged(_dedup_prompt(["Alice"], ["Bob"]), _EXISTING, "RENAMED")
        client = ForgetAwareLLMClient(_inner({"duplicate_facts": [0]}))

        answer = await client.generate_response(prompt, EdgeDuplicate)

        assert answer == {"duplicate_facts": [0]}
        assert "RENAMED" in caplog.text


class TestOffForgotten:
    def test_a_forgotten_edge_is_neither_duplicate_nor_contradiction(self) -> None:
        answer = {"duplicate_facts": [0, 1], "contradicted_facts": [1, 3]}

        assert off_forgotten(answer, {0, 3}) == {
            "duplicate_facts": [1],
            "contradicted_facts": [1],
        }

    def test_unreadable_lists_keep_the_new_fact_new(self) -> None:
        answer = {"duplicate_facts": [1], "contradicted_facts": [2]}

        assert off_forgotten(answer, None) == {
            "duplicate_facts": [],
            "contradicted_facts": [],
        }

    def test_a_malformed_answer_is_left_for_graphiti_to_reject(self) -> None:
        assert off_forgotten({"duplicate_facts": "x"}, {0}) == {"duplicate_facts": "x"}


class TestForgetAwareLLMClient:
    @pytest.mark.asyncio
    async def test_edge_dedup_answers_are_kept_off_forgotten_edges(self) -> None:
        inner = _inner({"duplicate_facts": [0], "contradicted_facts": [1]})
        client = ForgetAwareLLMClient(inner)
        prompt = _dedup_prompt([FORGOTTEN_FACT], ["Bob leads Atlas"])

        answer = await client.generate_response(
            prompt,
            EdgeDuplicate,
            model_size=ModelSize.small,
            prompt_name="dedupe_edges.resolve_edge",
        )

        assert answer == {"duplicate_facts": [], "contradicted_facts": [1]}
        inner.generate_response.assert_awaited_once_with(
            prompt,
            EdgeDuplicate,
            None,
            ModelSize.small,
            None,
            "dedupe_edges.resolve_edge",
            attribute_extraction=False,
        )

    @pytest.mark.asyncio
    async def test_every_other_call_passes_through(self) -> None:
        answer = {"edges": [], "note": FORGOTTEN_FACT}
        client = ForgetAwareLLMClient(_inner(answer))
        prompt = [Message(role="user", content=f"extract: {FORGOTTEN_FACT}")]

        assert await client.generate_response(prompt, ExtractedEdges) == answer

    def test_it_shares_the_inner_clients_tracer_and_token_counts(self) -> None:
        inner = _inner({})
        client = ForgetAwareLLMClient(inner)
        tracer = MagicMock()

        client.set_tracer(tracer)

        inner.set_tracer.assert_called_once_with(tracer)
        assert client.token_tracker is inner.token_tracker


def test_every_graphiti_client_is_built_forget_aware() -> None:
    inner = _inner({})
    with patch("graphiti_core.Graphiti") as graphiti:
        _build_graphiti(
            "user_abc",
            inner,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            graph_driver=MagicMock(),
        )

    wrapped = graphiti.call_args.kwargs["llm_client"]
    assert isinstance(wrapped, ForgetAwareLLMClient) and wrapped.inner is inner


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]


def _failing_parser(text: str) -> None:
    raise RuntimeError("parser bug")


_THEN = datetime(2026, 9, 27, tzinfo=timezone.utc)


def _edge(uuid: str, fact: str) -> EntityEdge:
    """An edge with its valid time known, so graphiti asks no timestamps."""
    return EntityEdge(
        uuid=uuid,
        group_id="user_abc",
        source_node_uuid="alice",
        target_node_uuid="atlas",
        created_at=_THEN,
        valid_at=_THEN,
        name="works_on",
        fact=fact,
    )


class TestAGuardThatHitsAnError:
    """Sentry's review of #14972: a candidate that reads as a literal but
    fails validation, or anything unexpected while the guard reads the
    prompt, names no edge, is logged at error once per shape and never
    stops graphiti's edge resolution."""

    @pytest.mark.parametrize(
        "old, new",
        [
            ("{'idx': 1, 'fact'", "{'idx': 'one', 'fact'"),
            ("{'idx': 0, 'fact'", "{'idx': 'zero', 'fact'"),
        ],
        ids=["a live candidate", "the forgotten candidate"],
    )
    @pytest.mark.asyncio
    async def test_a_candidate_that_fails_validation_names_no_edge(
        self, old: str, new: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = ForgetAwareLLMClient(_inner(dict(_NAMES_LIVE_EDGES)))
        prompt = _edited(_mixed_prompt(), old, new)

        first = await client.generate_response(prompt, EdgeDuplicate)
        logged = _errors(caplog)
        again = await client.generate_response(prompt, EdgeDuplicate)

        assert first == again == _NAMES_NOTHING
        assert logged and _errors(caplog) == logged, "once per shape"

    @pytest.mark.asyncio
    async def test_an_unexpected_error_while_reading_names_no_edge(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(recall_ingest, "_listed", _failing_parser)
        client = ForgetAwareLLMClient(_inner(dict(_NAMES_LIVE_EDGES)))

        for _ in range(2):
            answer = await client.generate_response(_mixed_prompt(), EdgeDuplicate)
            assert answer == _NAMES_NOTHING

        [error] = [r for r in caplog.records if r.levelname == "ERROR"]
        assert "raised RuntimeError" in error.getMessage()
        assert error.exc_info is not None, "with its traceback"

    @pytest.mark.asyncio
    async def test_graphitis_edge_resolution_still_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """graphiti's own resolution of a new statement, its model naming
        the forgotten edge a duplicate: the statement stays a new edge and
        the forgotten one is untouched, instead of the error ending
        ``add_episode``."""
        monkeypatch.setattr(recall_ingest, "_listed", _failing_parser)
        duplicate = {"duplicate_facts": [0], "contradicted_facts": []}
        client = ForgetAwareLLMClient(_inner(duplicate))
        forgotten = _edge("forgotten", FORGOTTEN_FACT)
        new = _edge("new", "Alice is assigned to work on the Atlas project")

        episode = EpisodicNode(
            name="s-2",
            group_id="user_abc",
            source=EpisodeType.text,
            source_description="chat",
            content=new.fact,
            created_at=_THEN,
            valid_at=_THEN,
        )

        resolved, invalidated, duplicates = await resolve_extracted_edge(
            client, new, [forgotten], [], episode
        )

        assert (resolved.uuid, invalidated, duplicates) == ("new", [], [])
        assert (forgotten.fact, forgotten.episodes) == (FORGOTTEN_FACT, [])
