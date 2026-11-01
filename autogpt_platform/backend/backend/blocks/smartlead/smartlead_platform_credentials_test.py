"""SmartLead campaign blocks refuse the platform's shared SmartLead key.

Campaigns live in the SmartLead account behind the key, so these blocks only
run on the user's own SmartLead key. The blocks are discovered, so one added to
``campaign.py`` later is covered too.
"""

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from backend.blocks._base import Block
from backend.blocks.smartlead import campaign
from backend.data.model import APIKeyCredentials
from backend.integrations.credentials_store import smartlead_credentials
from backend.util.exceptions import BlockExecutionError

CAMPAIGN_BLOCKS = [
    cls
    for _, cls in inspect.getmembers(campaign, inspect.isclass)
    if issubclass(cls, Block) and cls.__module__ == campaign.__name__
]
USER_OWNED = APIKeyCredentials(
    id="5d1e3c1a-7f1c-4a87-8a52-2a1c6f0b9e44",
    provider="smartlead",
    api_key=SecretStr("user-key"),
    title="My SmartLead key",
)


class ReachedVendor(BaseException):
    """Raised by the mocked SmartLead client."""


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(side_effect=ReachedVendor("SmartLeadClient"))
    monkeypatch.setattr(campaign, "SmartLeadClient", mock)
    return mock


def test_discovery_finds_the_campaign_blocks():
    assert {cls.__name__ for cls in CAMPAIGN_BLOCKS} >= {
        "CreateCampaignBlock",
        "AddLeadToCampaignBlock",
        "SaveCampaignSequencesBlock",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls", CAMPAIGN_BLOCKS, ids=lambda c: c.__name__)
async def test_campaign_block_refuses_platform_key(
    block_cls: type[Block], client: MagicMock
):
    with pytest.raises(BlockExecutionError, match="platform's shared SmartLead key"):
        await _run(block_cls(), smartlead_credentials)

    client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("block_cls", CAMPAIGN_BLOCKS, ids=lambda c: c.__name__)
async def test_campaign_block_runs_on_users_own_key(
    block_cls: type[Block], client: MagicMock
):
    with pytest.raises(ReachedVendor):
        await _run(block_cls(), USER_OWNED)

    client.assert_called_once_with(USER_OWNED.api_key.get_secret_value())


async def _run(block: Block, credentials: APIKeyCredentials) -> None:
    async for _ in block.run(_sample_input(block), credentials=credentials):
        pass


def _sample_input(block: Block) -> Any:
    values = {"name": "Campaign", "campaign_id": 1, "lead_list": [], "sequences": []}
    fields = block.input_schema.model_fields
    return block.input_schema.model_construct(
        **{name: value for name, value in values.items() if name in fields}
    )
