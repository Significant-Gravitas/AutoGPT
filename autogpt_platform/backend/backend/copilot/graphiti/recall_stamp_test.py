"""Recall stamps against a mocked driver: the one batched write each recall
hook makes, what it may touch, and that a failing stamp never fails a turn.
``recall_stamp_integration_test.py`` runs the same Cypher on FalkorDB."""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.dream import ratification as ratification_mod

from . import recall_stamp
from .recall import live_fact_predicate
from .recall_stamp import (
    RECALL_DEDUPE_INTERVAL,
    RecallStamp,
    parse_stamp,
    read_neighbour_stamps,
    read_recall_stamps,
    record_recall,
    stamp_fields,
    stamp_recalls,
    stamp_recalls_in_scope,
    stamp_time,
)
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-1")


def _driver(rows: list[dict] | Exception) -> AsyncMock:
    driver = AsyncMock()
    if isinstance(rows, Exception):
        driver.execute_query.side_effect = rows
    else:
        driver.execute_query.return_value = (rows, None, None)
    return driver


class TestTheStampWrite:
    @pytest.mark.asyncio
    async def test_one_batched_unwind_stamps_every_fact_once(self) -> None:
        driver = _driver([{"stamped": 2}])

        stamped = await stamp_recalls(driver, ["e1", "e2", "e1"], owner="user-1")

        assert stamped == 2
        driver.execute_query.assert_awaited_once()
        query = driver.execute_query.await_args.args[0]
        params = driver.execute_query.await_args.kwargs
        assert query.lstrip().startswith("UNWIND $uuids AS target_uuid")
        assert params["uuids"] == ["e1", "e2"], "a fact is stamped once per recall"
        # Edges written before the stamps start from zero, not NULL.
        assert "SET e.recall_count = coalesce(e.recall_count, 0) + 1" in query
        # The previous stamp shifts down before the new one lands.
        assert "e.prev_recalled_at = e.last_recalled_at" in query
        assert "e.last_recalled_at = $now" in query
        # Only the three usage properties: never the fact's envelope.
        written = query.split("SET", 1)[1].split("RETURN", 1)[0]
        for envelope in ("status", "expired_at", "forgotten_at", "fact ="):
            assert envelope not in written

    @pytest.mark.asyncio
    async def test_a_recall_within_the_dedupe_interval_is_not_counted(self) -> None:
        driver = _driver([{"stamped": 0}])
        before = datetime.now(timezone.utc)

        await stamp_recalls(driver, ["e1"], owner="user-1")

        query = driver.execute_query.await_args.args[0]
        params = driver.execute_query.await_args.kwargs
        assert (
            "(e.last_recalled_at IS NULL OR e.last_recalled_at < $dedupe_cutoff)"
            in query
        )
        now = parse_stamp(params["now"])
        cutoff = parse_stamp(params["dedupe_cutoff"])
        assert now is not None and cutoff is not None
        assert now - cutoff == RECALL_DEDUPE_INTERVAL
        assert now >= before
        # One width for every stamp, so the string comparison is by time.
        assert len(params["now"]) == len(params["dedupe_cutoff"])
        assert params["now"].endswith("+00:00")

    @pytest.mark.asyncio
    async def test_only_a_live_fact_is_stamped(self) -> None:
        """A forgotten fact (``forgotten_at``, ``retracted``, the legacy
        expired-without-reason shape) and an expired one all fail recall's
        live test, which the write carries."""
        driver = _driver([{"stamped": 0}])

        await stamp_recalls(driver, ["e1"], owner="user-1")

        query = driver.execute_query.await_args.args[0]
        assert f"e.uuid = target_uuid AND {live_fact_predicate('e')}" in query
        assert "e.forgotten_at IS NULL" in query
        assert "e.expired_at IS NULL" in query

    @pytest.mark.asyncio
    async def test_nothing_to_stamp_makes_no_query(self) -> None:
        driver = _driver([])

        assert await stamp_recalls(driver, [], owner="user-1") == 0
        driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failing_stamp_is_logged_and_never_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver(RuntimeError("falkordb down"))

        with caplog.at_level(logging.WARNING, logger=recall_stamp.__name__):
            assert await stamp_recalls(driver, ["e1"], owner="user-1") == 0

        assert "Recall stamp failed" in caplog.text

    @pytest.mark.asyncio
    async def test_an_unreadable_answer_counts_as_nothing_stamped(self) -> None:
        assert await stamp_recalls(_driver([]), ["e1"], owner="u") == 0
        assert await stamp_recalls(_driver([{"stamped": "2"}]), ["e1"], owner="u") == 0


class TestTheHooks:
    @pytest.mark.asyncio
    async def test_warm_context_stamps_every_retrieved_fact_in_one_write(
        self, mocker
    ) -> None:
        """The warm-context hook stamps every retrieved fact, not only the
        tentative ones it may promote: a long-lived active fact is the one
        whose use matters most."""
        mocker.patch.object(ratification_mod, "record_memory_hit", AsyncMock())
        queries: list[tuple[str, dict]] = []

        async def execute(query: str, **params):
            queries.append((query, params))
            return ([{"stamped": 3}] if "recall_count" in query else [], None, None)

        driver = MagicMock()
        driver.close = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=execute)
        mocker.patch.object(ratification_mod, "open_driver", return_value=driver)

        await ratification_mod.try_ratify_on_hit(_SCOPE, ["e1", "e2", "e3"])

        stamps = [(q, p) for q, p in queries if "recall_count" in q]
        assert len(stamps) == 1
        assert stamps[0][1]["uuids"] == ["e1", "e2", "e3"]

    @pytest.mark.asyncio
    async def test_warm_context_still_promotes_when_the_stamp_fails(
        self, mocker
    ) -> None:
        mocker.patch.object(ratification_mod, "record_memory_hit", AsyncMock())

        async def execute(query: str, **params):
            if "recall_count" in query:
                raise RuntimeError("stamp exploded")
            return ([{"uuid": params.get("uuid")}], None, None)

        driver = MagicMock()
        driver.close = AsyncMock()
        driver.execute_query = AsyncMock(side_effect=execute)
        mocker.patch.object(ratification_mod, "open_driver", return_value=driver)

        assert await ratification_mod.try_ratify_on_hit(_SCOPE, ["e1"]) == 1

    @pytest.mark.asyncio
    async def test_memory_search_counts_the_hit_and_stamps_its_scope(
        self, mocker
    ) -> None:
        """An expert's recall stamps the expert's own graph."""
        record_hit = mocker.patch.object(recall_stamp, "record_hit", AsyncMock())
        driver = _driver([{"stamped": 2}])
        driver.close = AsyncMock()
        open_driver = mocker.patch.object(
            recall_stamp, "open_driver", return_value=driver
        )
        expert = MemoryScope.for_expert("user-1", "expert-1")

        await record_recall(expert, ["e1", "e2"])

        record_hit.assert_awaited_once_with(expert, ["e1", "e2"])
        open_driver.assert_called_once_with(expert)
        assert driver.execute_query.await_args.kwargs["uuids"] == ["e1", "e2"]
        driver.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_memory_search_never_raises_when_the_graph_is_down(
        self, mocker
    ) -> None:
        record_hit = mocker.patch.object(recall_stamp, "record_hit", AsyncMock())
        mocker.patch.object(
            recall_stamp, "open_driver", side_effect=RuntimeError("no graph")
        )

        await record_recall(_SCOPE, ["e1"])

        record_hit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_close_that_fails_is_swallowed(self, mocker) -> None:
        driver = _driver([{"stamped": 1}])
        driver.close = AsyncMock(side_effect=RuntimeError("socket gone"))
        mocker.patch.object(recall_stamp, "open_driver", return_value=driver)

        assert await stamp_recalls_in_scope(_SCOPE, ["e1"]) == 1


class TestReadingTheStamps:
    @pytest.mark.asyncio
    async def test_the_targets_are_read_live_and_in_the_pass_graph_only(
        self,
    ) -> None:
        driver = _driver(
            [
                {
                    "uuid": "e1",
                    "recall_count": 4,
                    "last_recalled_at": "2026-09-27T10:00:00.000000+00:00",
                    "prev_recalled_at": None,
                }
            ]
        )

        stamps = await read_recall_stamps(driver, "user_g", ["e1", "e2", "e1"])

        assert stamps == [
            RecallStamp(
                uuid="e1",
                recall_count=4,
                last_recalled_at="2026-09-27T10:00:00.000000+00:00",
            )
        ]
        query = driver.execute_query.await_args.args[0]
        params = driver.execute_query.await_args.kwargs
        assert params == {"uuids": ["e1", "e2"], "group_id": "user_g"}
        assert "e.group_id = $group_id" in query
        assert live_fact_predicate("e") in query
        assert "toString(e.last_recalled_at) AS last_recalled_at" in query

    @pytest.mark.asyncio
    async def test_the_neighbours_read_matches_the_invalidations_neighbours(
        self,
    ) -> None:
        driver = _driver([{"uuid": "e1"}, {"uuid": "e2", "recall_count": 1}])

        stamps = await read_neighbour_stamps(driver, "user_g", "entity-1")

        assert [s.uuid for s in stamps or []] == ["e1", "e2"]
        query = driver.execute_query.await_args.args[0]
        assert "(n:Entity {uuid: $entity_uuid, group_id: $group_id})" in query
        assert "-[e:RELATES_TO]-()" in query, "both directions, single hop"
        assert f"WHERE {live_fact_predicate('e')}" in query
        assert "RETURN DISTINCT e.uuid AS uuid" in query

    @pytest.mark.asyncio
    async def test_a_failed_read_says_so_rather_than_no_stamps(self) -> None:
        driver = _driver(RuntimeError("falkordb down"))

        assert await read_recall_stamps(driver, "user_g", ["e1"]) is None
        assert await read_neighbour_stamps(driver, "user_g", "entity-1") is None

    @pytest.mark.asyncio
    async def test_no_targets_need_no_read(self) -> None:
        driver = _driver([])

        assert await read_recall_stamps(driver, "user_g", []) == []
        driver.execute_query.assert_not_awaited()


class TestStampValues:
    def test_a_stamp_time_has_one_width_in_utc(self) -> None:
        whole_second = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        offset = datetime(2026, 9, 1, 14, 0, 0, 5, tzinfo=timezone(timedelta(hours=2)))

        assert stamp_time(whole_second) == "2026-09-01T12:00:00.000000+00:00"
        assert stamp_time(offset) == "2026-09-01T12:00:00.000005+00:00"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("2026-09-01T12:00:00.000000+00:00", datetime(2026, 9, 1, 12)),
            ("2026-09-01T12:00:00Z", datetime(2026, 9, 1, 12)),
            ("2026-09-01T12:00:00", datetime(2026, 9, 1, 12)),
            ("not a time", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse_stamp(self, raw: str | None, expected: datetime | None) -> None:
        parsed = parse_stamp(raw)
        if expected is None:
            assert parsed is None
        else:
            assert parsed == expected.replace(tzinfo=timezone.utc)

    def test_a_row_without_stamps_reads_as_never_recalled(self) -> None:
        assert stamp_fields({}) == {
            "recall_count": None,
            "last_recalled_at": None,
            "prev_recalled_at": None,
        }

    def test_malformed_stamp_values_read_as_absent(self) -> None:
        assert stamp_fields(
            {"recall_count": "7", "last_recalled_at": 5, "prev_recalled_at": ""}
        ) == {"recall_count": None, "last_recalled_at": None, "prev_recalled_at": None}
        assert stamp_fields({"recall_count": 3.0})["recall_count"] == 3
        assert stamp_fields({"recall_count": True})["recall_count"] is None
