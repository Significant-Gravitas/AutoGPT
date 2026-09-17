"""Block-level `run()` tests for behaviour the standard harness can't reach.

The harness exercises one canned input per block. These cover the paths it
misses: the optional markdown report, and on-call de-duplication across teams,
which is the whole point of that block and can't be seen with a single-team
fixture.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.blocks.allquiet._api import AllQuietClient
from backend.blocks.allquiet._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.allquiet._testdata import (
    TEST_INCIDENT,
    TEST_MARKDOWN,
    TEST_SHIFT,
    TEST_USER,
)
from backend.blocks.allquiet._types import (
    AllQuietEntity,
    AllQuietUser,
    IncidentIntent,
    IncidentSeverity,
    OnCallAvailability,
    OnCallShift,
)
from backend.blocks.allquiet.incident_search import AllQuietGetIncidentBlock
from backend.blocks.allquiet.incidents import (
    AllQuietCreateIncidentBlock,
    AllQuietUpdateIncidentBlock,
)
from backend.blocks.allquiet.on_call import AllQuietGetOnCallBlock


async def _run(block, **inputs) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
        credentials=TEST_CREDENTIALS,
    ):
        collected.setdefault(name, value)
    return collected


class TestGetIncidentMarkdown:
    async def test_omits_markdown_by_default(self, monkeypatch: pytest.MonkeyPatch):
        block = AllQuietGetIncidentBlock()
        monkeypatch.setattr(
            block,
            "get_incident",
            AsyncMock(return_value=(TEST_INCIDENT, "")),
        )

        out = await _run(block, incident_id=TEST_INCIDENT.id)

        assert "markdown" not in out

    async def test_emits_markdown_when_requested(self, monkeypatch: pytest.MonkeyPatch):
        block = AllQuietGetIncidentBlock()
        monkeypatch.setattr(
            block,
            "get_incident",
            AsyncMock(return_value=(TEST_INCIDENT, TEST_MARKDOWN)),
        )

        out = await _run(block, incident_id=TEST_INCIDENT.id, include_markdown=True)

        assert out["markdown"] == TEST_MARKDOWN

    async def test_passes_the_flag_through_to_the_fetcher(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = AllQuietGetIncidentBlock()
        fetch = AsyncMock(return_value=(TEST_INCIDENT, TEST_MARKDOWN))
        monkeypatch.setattr(block, "get_incident", fetch)

        await _run(block, incident_id=TEST_INCIDENT.id, include_markdown=True)

        assert fetch.await_args.args[2].include_markdown is True


def _shift_for(user: AllQuietUser, team_name: str, team_id: str) -> OnCallShift:
    return OnCallShift(
        user=user,
        team=AllQuietEntity(id=team_id, displayName=team_name),
        availabilities=[OnCallAvailability(tier=1, isOnline=True, fillUp=False)],
    )


class TestOnCallDeduplication:
    async def test_a_user_on_two_teams_is_emitted_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Someone covering two rotations must not be paged twice.
        block = AllQuietGetOnCallBlock()
        shifts = [
            _shift_for(TEST_USER, "Platform", "team-1"),
            _shift_for(TEST_USER, "Website", "team-2"),
        ]
        monkeypatch.setattr(block, "get_on_call", AsyncMock(return_value=shifts))

        out = await _run(block)

        assert len(out["shifts"]) == 2, "every shift is still reported"
        assert out["user_ids"] == [TEST_USER.id]
        assert out["emails"] == [TEST_USER.email]
        assert len(out["users"]) == 1

    async def test_distinct_users_are_all_kept(self, monkeypatch: pytest.MonkeyPatch):
        block = AllQuietGetOnCallBlock()
        other = AllQuietUser(
            id="c0ffee00-0000-4000-8000-000000000009",
            displayName="Grace Hopper",
            email="grace@example.com",
        )
        monkeypatch.setattr(
            block,
            "get_on_call",
            AsyncMock(
                return_value=[
                    _shift_for(TEST_USER, "Platform", "team-1"),
                    _shift_for(other, "Platform", "team-1"),
                ]
            ),
        )

        out = await _run(block)

        assert out["user_ids"] == [TEST_USER.id, other.id]
        assert out["has_coverage"] is True

    async def test_reports_no_coverage_when_nobody_is_on_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = AllQuietGetOnCallBlock()
        monkeypatch.setattr(block, "get_on_call", AsyncMock(return_value=[]))

        out = await _run(block)

        assert out["has_coverage"] is False
        assert out["users"] == []
        assert out["emails"] == []

    async def test_skips_shifts_with_no_user(self, monkeypatch: pytest.MonkeyPatch):
        block = AllQuietGetOnCallBlock()
        headless = OnCallShift(
            user=None,
            team=AllQuietEntity(id="team-1", displayName="Platform"),
            availabilities=[],
        )
        monkeypatch.setattr(
            block, "get_on_call", AsyncMock(return_value=[headless, TEST_SHIFT])
        )

        out = await _run(block)

        assert out["user_ids"] == ["b7c8d9e0-0000-4000-8000-000000000002"]


class TestBlockErrorPaths:
    """A client failure must surface on the block's declared `error` output.

    The client is proven to raise on 4xx/5xx elsewhere; this wires that raise
    to the block boundary a graph actually sees.
    """

    async def test_get_incident_surfaces_a_client_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = AllQuietGetIncidentBlock()
        monkeypatch.setattr(
            block,
            "get_incident",
            AsyncMock(side_effect=RuntimeError("All Quiet API error 404: Not found")),
        )

        with pytest.raises(RuntimeError, match="Not found"):
            await _run(block, incident_id="missing")

    async def test_create_incident_surfaces_a_client_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = AllQuietCreateIncidentBlock()
        monkeypatch.setattr(
            block,
            "create_incident",
            AsyncMock(
                side_effect=RuntimeError(
                    "All Quiet rejected the API key (401). Check the key is valid"
                )
            ),
        )

        with pytest.raises(RuntimeError, match="rejected the API key"):
            await _run(block, title="anything")

    async def test_on_call_surfaces_a_client_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        block = AllQuietGetOnCallBlock()
        monkeypatch.setattr(
            block, "get_on_call", AsyncMock(side_effect=RuntimeError("upstream down"))
        )

        with pytest.raises(RuntimeError, match="upstream down"):
            await _run(block)

    def test_every_block_declares_an_error_output(self):
        # The platform routes a raised exception to this output, so a block
        # without it would swallow failures in the builder.
        for block_cls in (
            AllQuietGetIncidentBlock,
            AllQuietCreateIncidentBlock,
            AllQuietGetOnCallBlock,
        ):
            assert "error" in block_cls().output_schema.model_fields


class TestUpdateIncidentReadBack:
    """`update_incident` applies an intent then re-reads; both halves must run.

    The block test mocks `update_incident` wholesale, so removing the re-read
    would keep that green. These mock the two client calls separately.
    """

    async def test_applies_the_intent_then_re_reads_the_incident(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        append = AsyncMock()
        get = AsyncMock(return_value=TEST_INCIDENT)
        monkeypatch.setattr(AllQuietClient, "append_intent", append)
        monkeypatch.setattr(AllQuietClient, "get_incident", get)

        block = AllQuietUpdateIncidentBlock()
        out = await _run(
            block,
            incident_id=TEST_INCIDENT.id,
            intent=IncidentIntent.RESOLVED,
            message="fixed",
        )

        assert append.await_count == 1, "the intent must actually be applied"
        assert append.await_args.args[0] == TEST_INCIDENT.id
        assert append.await_args.kwargs["intent"] == IncidentIntent.RESOLVED.value
        assert append.await_args.kwargs["message"] == "fixed"

        assert get.await_count == 1, "the incident must be re-read afterwards"
        assert out["incident"] == TEST_INCIDENT
        assert out["allowed_intents"] == TEST_INCIDENT.allowed_intents

    async def test_passes_an_optional_severity_change_through(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        append = AsyncMock()
        monkeypatch.setattr(AllQuietClient, "append_intent", append)
        monkeypatch.setattr(
            AllQuietClient, "get_incident", AsyncMock(return_value=TEST_INCIDENT)
        )

        await _run(
            AllQuietUpdateIncidentBlock(),
            incident_id=TEST_INCIDENT.id,
            intent=IncidentIntent.COMMENTED,
            severity=IncidentSeverity.CRITICAL,
        )

        assert append.await_args.kwargs["severity"] == IncidentSeverity.CRITICAL
