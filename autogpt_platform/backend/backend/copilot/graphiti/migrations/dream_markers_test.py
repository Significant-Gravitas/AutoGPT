"""Unit tests for the operator's dream marker command: what it lists, what a
dry run would resolve, and what ``--apply`` resolves under the graph's write
lock (a marker whose episode graphiti saved is completed, any other deleted
only while no saved episode has its uuid). The live run is in
``recall_marker_integration_test.py``.
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.recall_reconcile import (
    DROP_UNLANDED_QUERY,
    KEPT,
    LANDED,
    MARKERS_QUERY,
    UNLANDED,
)
from backend.copilot.graphiti.scope import write_lock_key

from . import dream_markers

_NOW = datetime.now(timezone.utc)


def _row(
    uuid: str, rank: int, *, state: str = "pending", saved: bool = False, hours=1
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "episode": f"episode-{uuid}",
        "facts": ["f1", "f2"],
        "episodes": ["ep1"],
        "created_at": (_NOW - timedelta(hours=hours)).isoformat(),
        "state": state,
        "saved": saved,
        "rank": rank,
    }


# A write that landed, one crashed mid-save (its episode saved), one that
# never landed and one expired.
_ROWS = [
    _row("landed", LANDED, saved=True),
    _row("partial", UNLANDED, saved=True),
    _row("young", UNLANDED),
    _row("old", KEPT, state="expired", hours=30),
]


def _driver(*results: list[dict[str, Any]]) -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[(r, [], None) for r in results])
    driver.close = AsyncMock()
    return driver


class TestList:
    @pytest.mark.asyncio
    async def test_lists_every_marker_with_its_state_age_and_landing(self) -> None:
        unreadable = {**_ROWS[3], "created_at": "long ago"}
        driver = _driver([*_ROWS[:3], unreadable])

        markers = await dream_markers.list_markers(driver, "user_a")

        read = driver.execute_query.await_args_list[0]
        assert read.args[0] == MARKERS_QUERY
        assert read.kwargs == {"group_id": "user_a", "limit": dream_markers.LIST_LIMIT}
        assert [(m.uuid, m.state, m.saved, m.landed) for m in markers] == [
            ("landed", "pending", True, True),
            ("partial", "pending", True, False),
            ("young", "pending", False, False),
            ("old", "expired", False, False),
        ]
        assert markers[0].age_hours == pytest.approx(1, abs=0.1)
        assert markers[3].age_hours is None
        assert (markers[0].facts, markers[0].episodes) == (2, 1)
        assert markers[0].episode == "episode-landed"


class TestResolve:
    @pytest.mark.asyncio
    async def test_a_dry_run_counts_what_it_would_do_and_writes_nothing(
        self,
    ) -> None:
        driver = _driver(_ROWS)
        wanted = dream_markers.Selection(uuids={"landed", "young"}, expired=True)

        found = await dream_markers.resolve_graph(driver, "user_a", wanted, apply=False)

        assert found == dream_markers.Resolved(completed=1, deleted=2)
        assert [c.args[0] for c in driver.execute_query.await_args_list] == [
            MARKERS_QUERY
        ]

    @pytest.mark.asyncio
    async def test_apply_completes_the_saved_and_deletes_the_rest_under_the_lock(
        self, lock_redis: FakeRedis
    ) -> None:
        """A write whose episode graphiti saved, even in part, is completed
        as reconcile would; the rest go only while no saved episode has
        their uuid (the second's was saved since the read)."""
        driver = _driver(_ROWS, [{"dropped": 1}], [{"dropped": 0}])
        held: list[bool] = []

        async def complete(driver_, graph: str, row: dict[str, Any]) -> bool:
            held.append(write_lock_key(graph) in lock_redis.values)
            return row["uuid"] == "landed"

        wanted = dream_markers.Selection(
            uuids={"landed", "partial", "young"}, expired=True
        )
        with patch.object(dream_markers, "complete", complete):
            found = await dream_markers.resolve_graph(
                driver, "user_a", wanted, apply=True
            )

        assert found == dream_markers.Resolved(completed=1, deleted=1, kept=2)
        assert held == [True, True]
        drops = [
            c.kwargs
            for c in driver.execute_query.await_args_list
            if c.args[0] == DROP_UNLANDED_QUERY
        ]
        assert drops == [{"uuid": "young"}, {"uuid": "old"}]

    @pytest.mark.asyncio
    async def test_only_the_selected_markers_are_touched(self) -> None:
        driver = _driver(_ROWS, [{"dropped": 1}])

        found = await dream_markers.resolve_graph(
            driver,
            "user_a",
            dream_markers.Selection(expired=True),
            apply=True,
        )

        assert found == dream_markers.Resolved(deleted=1)
        assert driver.execute_query.await_args_list[1].kwargs == {"uuid": "old"}

    @pytest.mark.asyncio
    async def test_a_busy_graph_is_skipped_unwritten(
        self, lock_redis: FakeRedis
    ) -> None:
        lock_redis.values[write_lock_key("user_a")] = "an ingestion's token"
        driver = _driver(_ROWS)

        with patch.object(dream_markers, "MARKERS_LOCK_WAIT_SECONDS", 0):
            found = await dream_markers.resolve_graph(
                driver, "user_a", dream_markers.Selection(expired=True), apply=True
            )

        assert found == dream_markers.Resolved(busy=1)
        driver.execute_query.assert_not_awaited()


class TestCommandLine:
    @pytest.mark.asyncio
    async def test_lists_then_resolves_and_exits_1_while_a_graph_failed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good, bad = _driver([_ROWS[2]], [_ROWS[2]], [{"dropped": 1}]), _driver()
        bad.execute_query.side_effect = RuntimeError("down")
        opened = MagicMock(side_effect=[good, bad])
        args = dream_markers.parser().parse_args(
            ["--resolve", "young", "--resolve", "other", "--apply"]
        )

        with (
            patch.object(dream_markers, "open_graph_driver", opened),
            patch.object(
                dream_markers,
                "list_graph_names",
                AsyncMock(return_value=["user_b", "other", "user_a"]),
            ),
        ):
            code = await dream_markers.main(args)

        out = capsys.readouterr().out
        assert code == 1
        assert [c.args[0] for c in opened.call_args_list] == ["user_a", "user_b"]
        assert '"uuid":"young"' in out and '"graph":"user_a"' in out
        assert "resolved: 0 completed, 1 deleted; 0 kept" in out
        assert "skipped 0 busy and 1 failed graphs" in out
        good.close.assert_awaited_once()
        bad.close.assert_awaited_once()

    def test_a_bare_run_only_lists(self) -> None:
        args = dream_markers.parser().parse_args([])
        wanted = dream_markers.Selection(
            uuids=set(args.resolve or []), expired=args.resolve_expired
        )

        assert not wanted.any() and not args.apply
