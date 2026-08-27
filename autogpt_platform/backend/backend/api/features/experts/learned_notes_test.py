from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.experts import learned_notes


async def test_invalidate_learned_rule_requires_a_rule_id():
    with patch.object(learned_notes, "AutoGPTFalkorDriver") as driver:
        assert (
            await learned_notes.invalidate_learned_rule("user-1", "expert-1", None)
            is False
        )

    driver.assert_not_called()


async def test_invalidate_learned_rule_rejects_an_invalid_memory_group():
    with (
        patch.object(
            learned_notes,
            "derive_memory_group_id",
            side_effect=ValueError("invalid group"),
        ),
        patch.object(learned_notes, "AutoGPTFalkorDriver") as driver,
    ):
        result = await learned_notes.invalidate_learned_rule(
            "user-1", "expert-1", "rule-1"
        )

    assert result is False
    driver.assert_not_called()


@pytest.mark.parametrize(("demoted", "expected"), [(["rule-1"], True), ([], False)])
async def test_invalidate_learned_rule_returns_the_demotion_state_and_closes_driver(
    demoted: list[str], expected: bool
):
    driver = SimpleNamespace(close=AsyncMock())
    with (
        patch.object(
            learned_notes, "derive_memory_group_id", return_value="memory-group"
        ),
        patch.object(learned_notes, "AutoGPTFalkorDriver", return_value=driver),
        patch.object(
            learned_notes,
            "mark_edges_superseded",
            new_callable=AsyncMock,
            return_value=(demoted, []),
        ) as mark_edges_superseded,
    ):
        result = await learned_notes.invalidate_learned_rule(
            "user-1", "expert-1", "rule-1"
        )

    assert result is expected
    mark_edges_superseded.assert_awaited_once_with(
        driver,
        ["rule-1"],
        learned_notes.LEARNED_NOTE_DELETED_REASON,
        new_status="contradicted",
        user_id="user-1",
        group_id="memory-group",
    )
    driver.close.assert_awaited_once()


async def test_invalidate_learned_rule_handles_graph_failure_and_closes_driver():
    driver = SimpleNamespace(close=AsyncMock())
    with (
        patch.object(
            learned_notes, "derive_memory_group_id", return_value="memory-group"
        ),
        patch.object(learned_notes, "AutoGPTFalkorDriver", return_value=driver),
        patch.object(
            learned_notes,
            "mark_edges_superseded",
            new_callable=AsyncMock,
            side_effect=RuntimeError("graph unavailable"),
        ),
    ):
        result = await learned_notes.invalidate_learned_rule(
            "user-1", "expert-1", "rule-1"
        )

    assert result is False
    driver.close.assert_awaited_once()
