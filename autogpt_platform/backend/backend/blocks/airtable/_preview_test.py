"""What the Airtable blocks show before they change a user's base, and how
they ask for one."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockEffect
from backend.blocks.airtable._api import create_field
from backend.blocks.airtable.bases import AirtableCreateBaseBlock
from backend.blocks.airtable.records import (
    AirtableCreateRecordsBlock,
    AirtableGetRecordBlock,
    AirtableListRecordsBlock,
)
from backend.blocks.airtable.schema import (
    AirtableCreateFieldBlock,
    AirtableCreateTableBlock,
    AirtableListSchemaBlock,
    AirtableUpdateFieldBlock,
    AirtableUpdateTableBlock,
)
from backend.blocks.airtable.triggers import AirtableWebhookTriggerBlock
from backend.data.model import APIKeyCredentials

ORDER_FIELDS = [
    {"name": "Order number", "type": "singleLineText"},
    {"name": "Total", "type": "currency", "options": {"precision": 2}},
    {
        "name": "Status",
        "type": "singleSelect",
        "options": {"choices": [{"name": "New"}, {"name": "Shipped"}]},
    },
    {"name": "Shipped on", "type": "dateTime"},
]

STRUCTURE_BLOCKS = [
    AirtableCreateBaseBlock,
    AirtableCreateTableBlock,
    AirtableUpdateTableBlock,
    AirtableCreateFieldBlock,
    AirtableUpdateFieldBlock,
]


@pytest.mark.parametrize("block", STRUCTURE_BLOCKS, ids=lambda b: b.__name__)
def test_changing_the_shape_of_a_base_always_asks_first(block):
    """New or changed tables and fields land in a base the user already
    relies on, so no supervisor may wave them through unseen."""
    assert block().effect is BlockEffect.EXTERNAL


def test_a_new_table_is_shown_by_name_and_fields_in_plain_words():
    preview = AirtableCreateTableBlock().change_preview(
        {"base_id": "appX", "table_name": "Orders", "table_fields": ORDER_FIELDS}
    )

    assert preview is not None
    assert preview.ask == "Add a table to your Airtable:"
    assert preview.object == "Orders"
    assert [t.model_dump() for t in preview.tables] == [
        {
            "name": "Orders",
            "description": "",
            "fields": [
                {"name": "Order number", "kind": "Text", "detail": ""},
                {"name": "Total", "kind": "Money", "detail": "precision 2"},
                {"name": "Status", "kind": "Choice of New, Shipped", "detail": ""},
                {"name": "Shipped on", "kind": "Date and time", "detail": ""},
            ],
        }
    ]
    assert preview.covers == ["table_name", "table_fields"]


def test_a_new_base_shows_every_table_it_creates():
    preview = AirtableCreateBaseBlock().change_preview(
        {
            "workspace_id": "wspX",
            "name": "Shop",
            "tables": [
                {"name": "Orders", "fields": ORDER_FIELDS[:1]},
                {"name": "Customers", "fields": [{"name": "Email", "type": "email"}]},
            ],
        }
    )

    assert preview is not None
    assert preview.ask == "Create a new Airtable base:"
    assert preview.object == "Shop"
    assert [(t.name, [f.kind for f in t.fields]) for t in preview.tables] == [
        ("Orders", ["Text"]),
        ("Customers", ["Email"]),
    ]
    assert preview.covers == ["name", "tables"]


def test_a_new_field_is_shown_with_its_kind():
    preview = AirtableCreateFieldBlock().change_preview(
        {
            "base_id": "appX",
            "table_id": "tblX",
            "name": "Notes",
            "field_type": "multilineText",
        }
    )

    assert preview is not None
    assert preview.ask == "Add a field to your Airtable table:"
    assert preview.object == "Notes"
    assert [f.model_dump() for f in preview.tables[0].fields] == [
        {"name": "Notes", "kind": "Long text", "detail": ""}
    ]


@pytest.mark.parametrize(
    "block, inputs, ask, obj, covers",
    [
        (
            AirtableUpdateTableBlock,
            {"base_id": "appX", "table_id": "tblLeads", "table_name": "Clients"},
            "Rename Airtable table tblLeads to:",
            "Clients",
            ["table_id", "table_name"],
        ),
        (
            AirtableUpdateTableBlock,
            {"base_id": "appX", "table_id": "tblLeads", "table_description": "Hot"},
            "Change Airtable table tblLeads",
            None,
            ["table_id"],
        ),
        (
            AirtableUpdateFieldBlock,
            {
                "base_id": "appX",
                "table_id": "tblLeads",
                "field_id": "fldDue",
                "name": "Due",
            },
            "Rename field fldDue in Airtable table tblLeads to:",
            "Due",
            ["table_id", "field_id", "name"],
        ),
        (
            AirtableUpdateFieldBlock,
            {
                "base_id": "appX",
                "table_id": "tblLeads",
                "field_id": "fldDue",
                "description": "When it's owed",
            },
            "Change field fldDue in Airtable table tblLeads",
            None,
            ["table_id", "field_id"],
        ),
    ],
    ids=[
        "table rename",
        "table without a new name",
        "field rename",
        "field without a new name",
    ],
)
def test_a_change_says_what_it_changes_and_to_what(block, inputs, ask, obj, covers):
    """The card names the table or field that changes, which the headline
    hides from the arguments, and says "rename ... to" only when there is a new
    name, never ending on a colon with nothing after it."""
    preview = block().change_preview(inputs)

    assert preview is not None
    assert (preview.ask, preview.object) == (ask, obj)
    assert not (obj is None and preview.ask.endswith(":"))
    assert preview.covers == covers


def test_a_type_the_preview_does_not_know_still_reads_as_words():
    preview = AirtableCreateTableBlock().change_preview(
        {"table_name": "T", "table_fields": [{"name": "Score", "type": "someNewType"}]}
    )

    assert preview is not None
    assert preview.tables[0].fields[0].kind == "Some new type"


def test_inputs_the_model_got_wrong_still_preview_what_they_can():
    """The card opens on the model's raw call, before validation."""
    preview = AirtableCreateTableBlock().change_preview(
        {"table_name": "Orders", "table_fields": "not a list"}
    )

    assert preview is not None
    assert preview.object == "Orders"
    assert preview.tables[0].fields == []


BASE_ID_BLOCKS = [
    AirtableGetRecordBlock,
    AirtableCreateTableBlock,
    AirtableUpdateTableBlock,
    AirtableCreateFieldBlock,
    AirtableUpdateFieldBlock,
    AirtableListSchemaBlock,
    AirtableCreateRecordsBlock,
    AirtableListRecordsBlock,
    AirtableWebhookTriggerBlock,
]


# What Airtable lets each level do: changing tables or fields, or watching a
# base with a webhook, needs a base creator; writing records needs an editor.
NEEDED_ACCESS = {
    AirtableListSchemaBlock: "read",
    AirtableListRecordsBlock: "read",
    AirtableGetRecordBlock: "read",
    AirtableCreateRecordsBlock: "edit",
    AirtableCreateTableBlock: "create",
    AirtableUpdateTableBlock: "create",
    AirtableCreateFieldBlock: "create",
    AirtableUpdateFieldBlock: "create",
    AirtableWebhookTriggerBlock: "create",
}


@pytest.mark.parametrize("block", BASE_ID_BLOCKS, ids=lambda b: b.__name__)
def test_a_base_is_picked_from_the_account_never_typed(block):
    """Nobody knows a base id by heart: the field asks for a pick from the
    bases the connected account can reach, and only from those it may use
    for this step."""
    schema = block().input_schema.jsonschema()["properties"]["base_id"]

    assert schema["format"] == "airtable-base-picker"
    assert schema["resource_picker"] == {
        "provider": "airtable",
        "access": NEEDED_ACCESS[block],
    }
    assert schema["title"] == "Airtable base"


@pytest.mark.asyncio
async def test_a_new_field_is_made_with_the_options_the_card_showed():
    """The card shows a field's description and options; the call must send
    them, or the user approves one field and gets another."""
    block = AirtableCreateFieldBlock()
    credentials = APIKeyCredentials(
        provider="airtable", title="Airtable", api_key=SecretStr("k")
    )
    inputs = block.Input.model_validate(
        {
            "credentials": {
                "id": credentials.id,
                "provider": "airtable",
                "type": "api_key",
            },
            "base_id": "appX",
            "table_id": "tblX",
            "name": "Time spent",
            "field_type": "duration",
            "description": "Hours on the order",
            "options": {"durationFormat": "h:mm"},
        }
    )
    with patch(
        "backend.blocks.airtable.schema.create_field",
        AsyncMock(return_value={"id": "fld1"}),
    ) as create_field:
        [_ async for _ in block.run(inputs, credentials=credentials)]

    assert create_field.await_args.args[5:] == (
        "Hours on the order",
        {"durationFormat": "h:mm"},
    )


@pytest.mark.parametrize(
    "description, options, detail",
    [
        (
            None,
            {"choices": [{"name": "New", "color": "greenBright"}, {"name": "Done"}]},
            "New: color greenBright",
        ),
        (None, {"durationFormat": "h:mm"}, "duration format h:mm"),
        (
            "Who ordered it",
            {"linkedTableId": "tblCustomers", "prefersSingleRecordLink": True},
            "Who ordered it · linked table id tblCustomers · prefers single "
            "record link on",
        ),
    ],
    ids=["choice colours", "a format", "a note and nested settings"],
)
def test_a_new_field_shows_every_setting_it_is_made_with(description, options, detail):
    """Creating one field on its own binds its note and every option, so the
    card says all of them; the raw arguments it covers are not listed again."""
    preview = AirtableCreateFieldBlock().change_preview(
        {
            "name": "Status",
            "field_type": "singleSelect",
            "description": description,
            "options": options,
        }
    )

    assert preview is not None
    [field] = preview.tables[0].fields
    assert field.detail == detail
    assert set(preview.covers) == {"name", "field_type", "description", "options"}


def test_a_choice_field_can_be_created_with_its_choices():
    """Select choices are a list of objects, not strings."""
    inputs = AirtableCreateFieldBlock.Input.model_validate(
        {
            "credentials": {
                "id": "c1",
                "provider": "airtable",
                "type": "api_key",
            },
            "base_id": "appX",
            "table_id": "tblX",
            "name": "Status",
            "field_type": "singleSelect",
            "options": {"choices": [{"name": "New"}, {"name": "Shipped"}]},
        }
    )

    assert inputs.options == {"choices": [{"name": "New"}, {"name": "Shipped"}]}


@pytest.mark.asyncio
async def test_a_field_named_like_a_boolean_keeps_its_words():
    """Names, descriptions and choice names are text even when they read
    "true" or "false"; only option values were ever meant as booleans."""
    credentials = APIKeyCredentials(
        provider="airtable", title="Airtable", api_key=SecretStr("k")
    )
    response = MagicMock()
    response.json.return_value = {"id": "fld1"}
    with patch("backend.blocks.airtable._api.Requests") as requests:
        requests.return_value.post = AsyncMock(return_value=response)
        await create_field(
            credentials,
            "appX",
            "tblX",
            "singleSelect",
            "True",
            "false",
            {"choices": [{"name": "True"}, {"name": "False"}]},
        )

    assert requests.return_value.post.await_args.kwargs["json"] == {
        "type": "singleSelect",
        "name": "True",
        "description": "false",
        "options": {"choices": [{"name": "True"}, {"name": "False"}]},
    }


def test_field_settings_and_notes_are_on_the_card_in_words():
    """Approving a table binds every setting of its fields, so the card says
    them rather than hiding them behind the field's kind."""
    preview = AirtableCreateTableBlock().change_preview(
        {
            "table_name": "Orders",
            "table_fields": [
                {
                    "name": "Total",
                    "type": "currency",
                    "description": "Order total with tax",
                    "options": {"precision": 2, "symbol": "$"},
                },
                {
                    "name": "Status",
                    "type": "singleSelect",
                    "options": {"choices": [{"name": "New"}]},
                },
            ],
        }
    )

    assert preview is not None
    total, status = preview.tables[0].fields
    assert total.detail == "Order total with tax · precision 2 · symbol $"
    assert status.detail == ""


def test_a_choice_setting_beyond_its_name_is_on_the_card():
    preview = AirtableCreateTableBlock().change_preview(
        {
            "table_name": "Orders",
            "table_fields": [
                {
                    "name": "Status",
                    "type": "singleSelect",
                    "options": {
                        "choices": [
                            {"name": "New", "color": "greenBright"},
                            {"name": "Shipped"},
                        ]
                    },
                }
            ],
        }
    )

    assert preview is not None
    [status] = preview.tables[0].fields
    assert status.kind == "Choice of New, Shipped"
    assert status.detail == "New: color greenBright"
