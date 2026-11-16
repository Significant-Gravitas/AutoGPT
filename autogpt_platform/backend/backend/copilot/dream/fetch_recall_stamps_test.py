"""The recall stamps reach the dream pass and survive every hop its input
takes: the gather (``fetch.py``), the input bundle's JSON
(``input_bundle.py``), the batch path's Redis copy (``batch_submit.py``) and
the pass's durable record (``data/dream_pass_update.py``). A fact without
stamps, or a bundle written before them, reads as never recalled."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.recall_stamp import recall_stamp_columns
from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import DreamPassUpdate
from backend.data.dream_pass_update import transition_args

from . import fetch as fetch_mod
from .batch_submit import persist_input_bundle, read_input_bundle
from .fetch import DreamInput, FactRow
from .input_bundle import input_bundle_from_dict, input_bundle_to_dict

_LAST = "2026-09-27T09:30:00.000000+00:00"
_PREV = "2026-09-20T18:00:00.000000+00:00"


def _row(uuid: str, **stamps: object) -> dict:
    return {
        "uuid": uuid,
        "source": "Alice",
        "target": "Atlas",
        "name": "works_on",
        "fact": "Alice works on Atlas",
        "scope": "real:global",
        "confidence": 0.9,
        "status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
        **stamps,
    }


def _fact(uuid: str, **stamps: object) -> FactRow:
    return FactRow.model_validate(_row(uuid, **stamps))


def _bundle(*facts: FactRow) -> DreamInput:
    now = datetime(2026, 9, 28, 3, tzinfo=timezone.utc)
    return DreamInput(
        user_id="u1",
        group_id="user_u1",
        window_start=now,
        window_end=now,
        facts=list(facts),
        known_fact_uuids={f.uuid for f in facts},
    )


_USED = _fact("used", recall_count=3, last_recalled_at=_LAST, prev_recalled_at=_PREV)
_NEVER = _fact("never")


@pytest.mark.asyncio
async def test_the_fact_gather_reads_each_facts_stamps() -> None:
    driver = AsyncMock()
    driver.execute_query.return_value = (
        [
            _row(
                "used",
                recall_count=3,
                last_recalled_at=_LAST,
                prev_recalled_at=_PREV,
            ),
            _row("never", recall_count=None, last_recalled_at=None),
        ],
        None,
        None,
    )

    facts = await fetch_mod._fetch_active_facts(driver, "user_u1", 50)

    assert recall_stamp_columns("e") in driver.execute_query.await_args.args[0]
    assert facts == [_USED, _NEVER]
    assert (facts[1].recall_count, facts[1].last_recalled_at) == (None, None)


@pytest.mark.asyncio
async def test_gather_dream_input_hands_the_stamps_to_the_pass(mocker) -> None:
    async def execute(query: str, **params: object):
        if "RELATES_TO {group_id: $g}" in query:
            return ([_row("used", recall_count=3, last_recalled_at=_LAST)], [], None)
        return ([], [], None)

    driver = AsyncMock()
    driver.execute_query.side_effect = execute
    mocker.patch.object(fetch_mod, "open_driver", return_value=driver)
    mocker.patch.object(
        fetch_mod,
        "chat_db",
        return_value=SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[])),
    )

    bundle = await fetch_mod.gather_dream_input(MemoryScope.for_user("u1"))

    [fact] = bundle.facts
    assert (fact.recall_count, fact.last_recalled_at, fact.prev_recalled_at) == (
        3,
        _LAST,
        None,
    )


def test_the_input_bundle_round_trips_the_stamps() -> None:
    bundle = _bundle(_USED, _NEVER)

    restored = input_bundle_from_dict(
        json.loads(json.dumps(input_bundle_to_dict(bundle)))
    )

    assert restored.facts == [_USED, _NEVER]


def test_a_bundle_written_before_the_stamps_reads_as_never_recalled() -> None:
    written = input_bundle_to_dict(_bundle(_USED))
    for fact in written["facts"]:
        for key in ("recall_count", "last_recalled_at", "prev_recalled_at"):
            del fact[key]

    [fact] = input_bundle_from_dict(written).facts

    assert (fact.recall_count, fact.last_recalled_at, fact.prev_recalled_at) == (
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_the_batch_path_keeps_the_stamps_across_its_redis_copy() -> None:
    """The batch callbacks rebuild their prompts from this copy, hours
    after the gather."""
    await persist_input_bundle("p-stamps", _bundle(_USED, _NEVER))

    restored = await read_input_bundle("p-stamps")

    assert restored is not None
    assert restored.facts == [_USED, _NEVER]


def test_the_pass_record_keeps_the_stamps() -> None:
    args = transition_args("p1", DreamPassUpdate(input_bundle=_bundle(_USED)))
    [column] = [a for a in args if '"known_fact_uuids"' in str(a)]

    stored = input_bundle_from_dict(json.loads(column))

    assert stored.facts == [_USED]
