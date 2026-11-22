"""Relative dates, grouping, and the owner's field renames and type changes,
against a real Postgres."""

import uuid
from datetime import date

import pytest

from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

from . import db
from .model import (
    Aggregate,
    CollectionError,
    Condition,
    FieldSpec,
    FieldType,
    RowQuery,
    workflow_writer,
)
from .schema import FieldRename, SchemaChange, TypeChange

pytestmark = pytest.mark.asyncio(loop_scope="session")

NUDGE = workflow_writer("graph-nudge")
# Mid-month and mid-week, so no window crosses a boundary.
TODAY = date(2026, 11, 18)


@pytest.fixture
async def user_id(server: SpinTestServer) -> str:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"views-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Lena",
        }
    )
    return user.id


async def _follow_ups(user_id: str) -> None:
    await db.create_collection(
        user_id,
        "Follow-ups",
        [
            FieldSpec(name="what", type=FieldType.TEXT),
            FieldSpec(
                name="client",
                type=FieldType.SELECT,
                options=["Brightline", "Oko Studio"],
            ),
            FieldSpec(name="due", type=FieldType.DATE),
            FieldSpec(name="hours", type=FieldType.NUMBER),
        ],
    )
    await db.write_rows(
        user_id,
        "Follow-ups",
        [
            {
                "what": "Revised quote",
                "client": "Brightline",
                "due": "2026-11-18",
                "hours": 2,
            },
            {
                "what": "Kickoff call",
                "client": "Oko Studio",
                "due": "2026-11-20T09:00:00Z",
                "hours": 1,
            },
            {
                "what": "Invoice chase",
                "client": "Brightline",
                "due": "2026-11-11",
                "hours": 0.5,
            },
            {
                "what": "Year plan",
                "client": "Oko Studio",
                "due": "2026-12-02",
                "hours": 3,
            },
            {"what": "Old note", "due": "2026-10-30", "hours": 1},
        ],
        writer=NUDGE,
    )


@pytest.mark.parametrize(
    "op, value, expected",
    [
        ("today", None, ["Revised quote"]),
        ("next_days", 7, ["Kickoff call", "Revised quote"]),
        ("last_days", 7, ["Invoice chase", "Revised quote"]),
        ("this_month", None, ["Invoice chase", "Kickoff call", "Revised quote"]),
    ],
)
async def test_relative_dates_read_from_the_readers_day(
    user_id: str, op: str, value: int | None, expected: list[str]
):
    await _follow_ups(user_id)

    page = await db.query_rows(
        user_id,
        "Follow-ups",
        RowQuery(conditions=[Condition(field="due", op=op, value=value)], today=TODAY),
    )

    assert sorted(r.values["what"] for r in page.rows) == expected


async def test_a_relative_date_on_a_text_field_is_refused(user_id: str):
    await _follow_ups(user_id)

    with pytest.raises(CollectionError, match="date fields"):
        await db.query_rows(
            user_id,
            "Follow-ups",
            RowQuery(conditions=[Condition(field="what", op="today")]),
        )


async def test_rows_group_by_a_select_in_option_order_and_by_week_of_a_date(
    user_id: str,
):
    await _follow_ups(user_id)

    by_client = await db.group_rows(
        user_id, "Follow-ups", "client", Aggregate(op="sum", field="hours")
    )
    by_week = await db.group_rows(
        user_id, "Follow-ups", "due", Aggregate(op="count"), bucket="week"
    )
    this_month = await db.group_rows(
        user_id,
        "Follow-ups",
        "due",
        Aggregate(op="count"),
        bucket="month",
        conditions=[Condition(field="due", op="this_month")],
        today=TODAY,
    )

    assert [(g.label, g.value) for g in by_client] == [
        ("Brightline", 2.5),
        ("Oko Studio", 4.0),
        (None, 1.0),
    ]
    assert [(g.label, g.value) for g in by_week] == [
        ("2026-10-26", 1.0),
        ("2026-11-09", 1.0),
        ("2026-11-16", 2.0),
        ("2026-11-30", 1.0),
    ]
    assert [(g.label, g.value) for g in this_month] == [("2026-11", 3.0)]


async def test_a_renamed_field_keeps_its_rows_and_its_old_name_writes_to_it(
    user_id: str,
):
    await _follow_ups(user_id)
    found = await db.get_collection(user_id, "Follow-ups")
    assert found is not None
    row = (await db.query_rows(user_id, found.id, RowQuery(search="Kickoff"))).rows[0]
    await db.edit_row(user_id, found.id, row.id, {"hours": 1.5}, based_on=row.revision)

    renamed = await db.change_schema(
        user_id,
        found.id,
        SchemaChange(rename_fields=[FieldRename(field="hours", to="estimate")]),
    )
    await db.update_row(user_id, found.id, {"hours": 4}, writer=NUDGE, row_id=row.id)

    field = next(f for f in renamed.fields if f.name == "estimate")
    assert field.aliases == ["hours"]
    after = await db.get_row(user_id, found.id, row_id=row.id)
    assert after is not None
    assert after.values["estimate"] == 1.5 and "hours" not in after.values
    assert after.manual_fields == ["estimate"]
    assert after.pending == {"estimate": 4}


async def test_a_type_change_converts_every_value(user_id: str):
    await db.write_rows(
        user_id,
        "Weekly KPIs",
        [
            {"week": "2026-11-09", "orders": "4812"},
            {"week": "2026-11-16", "orders": "4790"},
        ],
        writer=NUDGE,
    )

    converted = await db.change_schema(
        user_id,
        "Weekly KPIs",
        SchemaChange(change_types=[TypeChange(field="orders", type=FieldType.NUMBER)]),
    )

    assert {f.name: f.type for f in converted.fields}["orders"] is FieldType.NUMBER
    page = await db.query_rows(user_id, "Weekly KPIs", RowQuery())
    assert sorted(r.values["orders"] for r in page.rows) == [4790, 4812]
    with pytest.raises(CollectionError, match="'lots' is not a number"):
        await db.write_rows(
            user_id,
            "Weekly KPIs",
            [{"week": "2026-11-30", "orders": "lots"}],
            writer=NUDGE,
        )


async def test_a_type_change_that_would_lose_a_value_is_refused(user_id: str):
    await db.write_rows(
        user_id,
        "Notes",
        [{"what": "42"}, {"what": "carrier strike"}],
        writer=NUDGE,
    )

    with pytest.raises(CollectionError, match="1 values in 'what' are not number"):
        await db.change_schema(
            user_id,
            "Notes",
            SchemaChange(
                change_types=[TypeChange(field="what", type=FieldType.NUMBER)]
            ),
        )

    page = await db.query_rows(user_id, "Notes", RowQuery())
    assert sorted(r.values["what"] for r in page.rows) == ["42", "carrier strike"]
