"""Verify Jev credential discovery and explicit BYOK cost registration."""

import pytest

from backend.blocks._base import BlockCostType
from backend.blocks.typesafe._config import typesafe
from backend.blocks.typesafe.ask_many import JevAskManyBlock
from backend.blocks.typesafe.choice import JevChoiceBlock
from backend.blocks.typesafe.filter import JevFilterBlock
from backend.blocks.typesafe.pick_best import JevPickBestBlock
from backend.blocks.typesafe.route import JevRouteBlock
from backend.blocks.typesafe.score import JevScoreBlock
from backend.blocks.typesafe.yes_no import JevYesNoBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.integrations.providers import ProviderName
from backend.sdk.cost_integration import get_block_costs
from backend.sdk.registry import AutoRegistry


def test_typesafe_provider_is_discoverable_without_a_shared_key():
    assert AutoRegistry.get_provider(ProviderName.TYPESAFE.value) is typesafe
    assert typesafe.supported_auth_types == {"api_key"}
    assert typesafe.default_credentials == []


@pytest.mark.parametrize(
    "block_type",
    [
        JevChoiceBlock,
        JevScoreBlock,
        JevAskManyBlock,
        JevRouteBlock,
        JevYesNoBlock,
        JevPickBestBlock,
        JevFilterBlock,
    ],
)
def test_every_jev_block_has_a_typed_key_and_explicit_cost_entry(block_type):
    block = block_type()
    credentials = block.input_schema.get_credentials_fields_info()["credentials"]
    assert credentials.provider == {ProviderName.TYPESAFE}
    assert credentials.supported_types == {"api_key"}
    costs = get_block_costs(block_type)
    assert BLOCK_COSTS[block_type] == costs
    assert len(costs) == 1
    assert costs[0].cost_type == BlockCostType.RUN
    assert costs[0].cost_amount == 0
