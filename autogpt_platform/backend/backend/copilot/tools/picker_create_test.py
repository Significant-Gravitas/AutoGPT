"""A picker card names the capability that creates what it picks (SECRT-3403)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import AnyBlockSchema
from backend.blocks.airtable.records import AirtableCreateRecordsBlock
from backend.blocks.google.docs import (
    GoogleDocsAppendPlainTextBlock,
    GoogleDocsCreateBlock,
)
from backend.blocks.google.sheets import (
    GoogleSheetsAppendRowBlock,
    GoogleSheetsCreateSpreadsheetBlock,
    GoogleSheetsReadBlock,
)
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.context import _current_permissions
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import SetupRequirementsResponse
from backend.copilot.tools.run_capability import _run_block
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput

USER = "user-picker-create"
ROW = ["12 Elm St", "3 bed", "$410,000"]
CREATE_SHEET = f"block:{GoogleSheetsCreateSpreadsheetBlock().id}"


@pytest.fixture(autouse=True)
def _no_database():
    """The card is built before anything is read from the account, so only
    the lookups ahead of it are stubbed."""
    reviews = MagicMock()
    reviews.get_pending_reviews_for_chat_session = AsyncMock(return_value=[])
    workspaces = MagicMock()
    workspaces.get_or_create_workspace = AsyncMock(
        return_value=SimpleNamespace(id="ws-1")
    )
    users = MagicMock()
    users.get_user_by_id = AsyncMock(return_value=SimpleNamespace(timezone="UTC"))
    with (
        patch("backend.copilot.tools.helpers.review_db", return_value=reviews),
        patch("backend.copilot.tools.helpers.workspace_db", return_value=workspaces),
        patch("backend.copilot.tools.helpers.user_db", return_value=users),
        patch(
            "backend.copilot.tools.helpers.match_credentials_to_requirements",
            _airtable_connected,
        ),
    ):
        yield


async def _airtable_connected(
    user_id: str, requirements: dict[str, CredentialsFieldInfo], *args, **kwargs
):
    """A resource picker is left off the card until an account can list it;
    Google's picker carries its own, so nothing is matched for Google."""
    matched = {
        field: CredentialsMetaInput(id="cred-1", provider="airtable", type="api_key")
        for field, info in requirements.items()
        if "airtable" in info.provider
    }
    return matched, []


@pytest.mark.parametrize(
    "block, inputs, creator",
    [
        pytest.param(
            GoogleSheetsAppendRowBlock(),
            {"spreadsheet": None, "row": ROW},
            CREATE_SHEET,
            id="append-row-names-create-spreadsheet",
        ),
        pytest.param(
            GoogleDocsAppendPlainTextBlock(),
            {"document": None, "text": "Viewing notes"},
            f"block:{GoogleDocsCreateBlock().id}",
            id="append-text-names-create-document",
        ),
        pytest.param(
            GoogleSheetsReadBlock(),
            {"spreadsheet": None, "range": "A1:C10"},
            None,
            id="a-read-creates-nothing",
        ),
        pytest.param(
            # What dev's block output hands back for a created sheet: the pick
            # is there, only its credential is not, so creating again is wrong.
            GoogleSheetsAppendRowBlock(),
            {"spreadsheet": {"id": "s-1", "credentials_id": "c-1"}, "row": ROW},
            None,
            id="a-picked-file-gets-no-hint",
        ),
        pytest.param(
            # Its create block outputs a bare base id and needs a workspace id
            # the user would have to find.
            AirtableCreateRecordsBlock(),
            {"base_id": None, "table_id_or_name": "Homes", "records": []},
            None,
            id="an-airtable-base-has-no-creator",
        ),
    ],
)
async def test_a_picker_card_names_what_creates_its_pick(
    block: AnyBlockSchema, inputs: dict[str, Any], creator: str | None
):
    card = await _card(block, inputs)

    if creator is None:
        assert card.create_hint is None
        assert card.setup_info.requirements["inputs"], "the card must offer a pick"
        return
    assert card.create_hint is not None
    assert card.create_hint.count("block:") == 1 and creator in card.create_hint
    # The card shows its message to the user; the hint is for the model.
    assert creator not in card.message
    assert '"create_hint"' in card.model_dump_json(exclude_none=True)


async def test_a_creator_the_session_may_not_run_is_not_named():
    blocked = CopilotPermissions(blocks=[GoogleSheetsCreateSpreadsheetBlock().id])
    token = _current_permissions.set(blocked)
    try:
        card = await _card(
            GoogleSheetsAppendRowBlock(), {"spreadsheet": None, "row": ROW}
        )
    finally:
        _current_permissions.reset(token)

    assert card.create_hint is None


async def _card(
    block: AnyBlockSchema, inputs: dict[str, Any]
) -> SetupRequirementsResponse:
    entry = get_registry().get(f"block:{block.id}")
    assert entry is not None
    result = await _run_block(
        entry,
        USER,
        make_session(USER),
        dict(inputs),
        validate_only=False,
        approved=True,
    )
    assert isinstance(result, SetupRequirementsResponse), result.message
    return result
