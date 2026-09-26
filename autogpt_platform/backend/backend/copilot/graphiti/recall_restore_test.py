"""Unit tests for ``recall_restore``: what putting a forgotten edge back
writes, and what it leaves alone, against a mock driver.

The live runs are ``recall_ingest_integration_test.py`` and
``recall_repair_integration_test.py``.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from . import recall_restore
from .recall import FORGOTTEN_FACT
from .recall_restore import (
    EXACT_FIELDS,
    FILL_FIELDS,
    ForgottenEdge,
    RestoreSpec,
    needs_restore,
    obligation,
    restore,
    spec_for,
)
from .recall_stash import ForgetRecord, read_forgets

_GROUP = "user_abc"
_WHEN = "2026-01-01T00:00:00+00:00"


def _record(**update: Any) -> ForgetRecord:
    return ForgetRecord(
        uuid="f1",
        forgotten_at=_WHEN,
        expired_at=_WHEN,
        fact_redacted="Alice works on Atlas",
        name_redacted="MemoryFact",
        stashed_at=datetime.now(timezone.utc).isoformat(),
        **update,
    )


def _state(**fields: Any) -> ForgottenEdge:
    """The edge as a finished forget leaves it, with ``fields`` changed."""
    base = {
        **_record().model_dump(include=set(EXACT_FIELDS)),
        "fact_redacted": "Alice works on Atlas",
        "name_redacted": "MemoryFact",
        "confidence": 0.9,
        "provenance": "session:s1#msg:1",
        "episodes": ["ep-old"],
    }
    return ForgottenEdge(
        uuid="f1", source="alice", target="atlas", fields=base | fields
    )


class TestSpec:
    def test_a_stashed_forget_is_put_back_as_it_set_it(self) -> None:
        snapshot = _state(provenance="session:s1#msg:1")

        spec = spec_for(_record(), snapshot)

        assert spec is not None
        assert spec.exact["forgotten_at"] == _WHEN
        assert (spec.exact["fact"], spec.exact["name"]) == (FORGOTTEN_FACT,) * 2
        assert spec.audit == {
            "fact_redacted": "Alice works on Atlas",
            "name_redacted": "MemoryFact",
        }
        assert spec.fill["provenance"] == "session:s1#msg:1"

    def test_without_a_stash_the_snapshot_is_the_record(self) -> None:
        spec = spec_for(None, _state(status="retracted"))

        assert spec is not None and spec.exact["status"] == "retracted"

    def test_neither_means_nothing_to_restore(self) -> None:
        assert spec_for(None, None) is None

    def test_a_forget_owns_its_marker_text_and_times_not_the_rest(self) -> None:
        assert set(EXACT_FIELDS) == {
            "forgotten_at",
            "status",
            "expiration_reason",
            "expired_at",
            "invalid_at",
            "valid_at",
            "fact",
            "name",
        }
        assert {"confidence", "provenance", "source_kind", "scope"} <= set(FILL_FIELDS)
        assert "status" not in FILL_FIELDS


class TestNeedsRestore:
    def _spec(self, **update: Any) -> RestoreSpec:
        spec = spec_for(_record(), _state())
        assert spec is not None
        return spec.model_copy(update=update)

    def test_an_intact_forget_needs_nothing(self) -> None:
        assert not needs_restore(_state(), self._spec())

    @pytest.mark.parametrize(
        "changed",
        [
            {"forgotten_at": None},
            {"status": "active"},
            {"fact": "Alice works on Atlas"},
            {"invalid_at": "2026-06-01T00:00:00+00:00"},
            {"fact_redacted": None},
            {"provenance": None},
            {"episodes": ["ep-old", "ep-new"]},
        ],
    )
    def test_graphiti_rewrites_need_a_restore(self, changed: dict[str, Any]) -> None:
        assert needs_restore(_state(**changed), self._spec(dropped=["ep-new"]))

    def test_another_writers_value_is_not_a_difference(self) -> None:
        """A field a forget does not own, changed rather than wiped."""
        assert not needs_restore(_state(confidence=0.92), self._spec())


class TestRestore:
    @pytest.mark.asyncio
    async def test_writes_the_forgets_fields_and_fills_the_rest(self) -> None:
        driver = AsyncMock()
        spec = spec_for(_record(), _state())
        assert spec is not None

        assert await restore(driver, _GROUP, spec)

        query = driver.execute_query.await_args.args[0]
        kwargs = driver.execute_query.await_args.kwargs
        assert "SET e += $exact," in query
        assert (
            "e.fact_redacted = coalesce(e.fact_redacted, $audit.fact_redacted)" in query
        )
        assert "e.confidence = coalesce(e.confidence, $fill.confidence)" in query
        assert "[x IN coalesce(e.episodes, []) WHERE NOT x IN $dropped]" in query
        assert "fact_embedding" not in query
        assert kwargs["exact"] == spec.exact and kwargs["uuid"] == "f1"

    @pytest.mark.asyncio
    async def test_one_failure_is_retried(self) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = [RuntimeError("blip"), None]
        spec = spec_for(_record(), None)
        assert spec is not None

        assert await restore(driver, _GROUP, spec)
        assert driver.execute_query.await_count == 2
        assert await read_forgets(_GROUP) == {}

    @pytest.mark.asyncio
    async def test_a_second_failure_leaves_an_obligation_in_the_stash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = AsyncMock()
        driver.execute_query.side_effect = RuntimeError("down")
        spec = spec_for(_record(dropped_episodes=["ep-new"]), _state())
        assert spec is not None

        assert not await restore(driver, _GROUP, spec)

        left = (await read_forgets(_GROUP))["f1"]
        assert (left.forgotten_at, left.fact_redacted) == (
            _WHEN,
            "Alice works on Atlas",
        )
        assert left.dropped_episodes == ["ep-new"]
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_an_obligation_from_a_legacy_forget_gets_a_marker(self) -> None:
        legacy = _state(forgotten_at=None, status="active", expiration_reason=None)
        spec = spec_for(None, legacy)
        assert spec is not None

        record = obligation(spec)

        assert record.forgotten_at == _WHEN, "the old forget's expiry"
        assert (record.status, record.expiration_reason) == ("active", "user_signal")


def test_the_restore_query_leaves_other_writers_alone() -> None:
    query = recall_restore._RESTORE_QUERY
    for field in FILL_FIELDS:
        assert f"e.{field} = coalesce(e.{field}, $fill.{field})" in query
    assert "e.episodes = $" not in query, "episodes are dropped, never replaced"
