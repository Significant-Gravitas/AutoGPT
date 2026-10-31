"""Verify nace credential discovery and explicit BYOK cost registration."""

import pytest

from backend.blocks._base import BlockCostType
from backend.blocks.nace._config import nace
from backend.blocks.nace.choice import NaceChoiceBlock
from backend.blocks.nace.score import NaceScoreBlock
from backend.blocks.nace.yes_no import NaceYesNoBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.integrations.providers import ProviderName
from backend.sdk.cost_integration import get_block_costs
from backend.sdk.registry import AutoRegistry


def test_nace_provider_is_discoverable_without_a_shared_key():
    assert AutoRegistry.get_provider(ProviderName.NACE.value) is nace
    assert nace.supported_auth_types == {"api_key"}
    assert nace.default_credentials == []


@pytest.mark.parametrize(
    "block_type", [NaceChoiceBlock, NaceScoreBlock, NaceYesNoBlock]
)
def test_every_drex_block_has_a_typed_key_and_explicit_cost_entry(block_type):
    block = block_type()
    credentials = block.input_schema.get_credentials_fields_info()["credentials"]
    assert credentials.provider == {ProviderName.NACE}
    assert credentials.supported_types == {"api_key"}
    costs = get_block_costs(block_type)
    assert BLOCK_COSTS[block_type] == costs
    assert len(costs) == 1
    assert costs[0].cost_type == BlockCostType.RUN
    assert costs[0].cost_amount == 0
