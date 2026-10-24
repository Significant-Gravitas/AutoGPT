"""A block input picked from the user's own account (an Airtable base) is
asked for in the setup card, never in chat as an id."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.airtable.schema import AirtableCreateTableBlock
from backend.copilot.tools.helpers import (
    get_picker_inputs_from_schema,
    is_picker_field,
    prepare_block_for_execution,
)
from backend.copilot.tools.models import ErrorResponse, SetupRequirementsResponse
from backend.data.model import CredentialsMetaInput

_USER = "picker-user"
_SESSION = "picker-session"

_SCHEMA = {
    "properties": {
        "base_id": {
            "type": "string",
            "title": "Airtable base",
            "format": "airtable-base-picker",
            "resource_picker": {"provider": "airtable"},
        },
        "table_name": {"type": "string"},
    },
    "required": ["base_id", "table_name"],
}


def test_a_resource_picker_is_a_picker_field():
    assert is_picker_field({"resource_picker": {"provider": "airtable"}})


def test_the_picker_lists_with_the_account_the_run_will_use():
    inputs = get_picker_inputs_from_schema(
        _SCHEMA, picker_credentials={"airtable": "cred-1"}
    )

    assert [i["name"] for i in inputs] == ["base_id"]
    assert inputs[0]["resource_picker"] == {
        "provider": "airtable",
        "credentials_id": "cred-1",
    }


def test_the_picker_waits_until_the_account_is_connected():
    """With no account there is nothing to list: the card signs in first and
    the picker follows when the call runs again."""
    assert get_picker_inputs_from_schema(_SCHEMA) == []
    assert (
        get_picker_inputs_from_schema(_SCHEMA, picker_credentials={"google": "g"}) == []
    )


def _airtable_creds() -> CredentialsMetaInput:
    return CredentialsMetaInput(
        id="cred-airtable", provider="airtable", type="oauth2", title="Airtable"
    )


async def _prepare(input_data: dict):
    session = MagicMock()
    session.session_id = _SESSION
    session.expert_id = None
    with (
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({"credentials": _airtable_creds()}, [])),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            AsyncMock(side_effect=lambda d, *a, **kw: d),
        ),
    ):
        return await prepare_block_for_execution(
            block_id=AirtableCreateTableBlock().id,
            input_data=input_data,
            user_id=_USER,
            session=session,
            session_id=_SESSION,
            dry_run=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, ""])
async def test_a_base_left_empty_as_the_prompt_says_opens_the_picker(empty):
    """The shared prompt tells the model to pass a picker input as null; a
    null must open the card, not count as a base."""
    result = await _prepare({"table_name": "Orders", "base_id": empty})

    assert isinstance(result, SetupRequirementsResponse)
    [picker] = result.setup_info.requirements["inputs"]
    assert picker["name"] == "base_id"


@pytest.mark.asyncio
async def test_a_missing_base_opens_the_picker_instead_of_asking_in_chat():
    """Before: the call came back as 'missing workspace_id/base_id' and the
    model asked the user to dig an id out of a URL."""
    session = MagicMock()
    session.session_id = _SESSION
    session.expert_id = None
    with (
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({"credentials": _airtable_creds()}, [])),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            AsyncMock(side_effect=lambda d, *a, **kw: d),
        ),
    ):
        result = await prepare_block_for_execution(
            block_id=AirtableCreateTableBlock().id,
            input_data={"table_name": "Orders"},
            user_id=_USER,
            session=session,
            session_id=_SESSION,
            dry_run=False,
        )

    assert isinstance(result, SetupRequirementsResponse)
    [picker] = result.setup_info.requirements["inputs"]
    assert picker["name"] == "base_id"
    assert picker["resource_picker"]["credentials_id"] == "cred-airtable"
    assert result.setup_info.user_readiness.has_all_credentials
    assert result.message == (
        "Choose your Airtable base in the card below. The step runs again with "
        "your choice."
    )
    assert "base_id" not in result.message


@pytest.mark.asyncio
async def test_an_unwatched_run_says_the_base_must_be_picked_in_a_chat():
    """A schedule or automation has nobody to answer a card, so the step
    reports what is missing instead of leaving a card nobody sees."""
    with patch("backend.copilot.tools.helpers.is_unattended_turn", return_value=True):
        result = await _prepare({"table_name": "Orders"})

    assert isinstance(result, ErrorResponse)
    assert "Airtable base" in result.message
    assert "nobody is watching" in result.message
