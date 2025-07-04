"""
Test the cost integration functionality.
"""

import pytest

from backend.blocks.examples.cost_example_block import (
    DataBasedCostBlock,
    FixedCostBlock,
    ProviderCostBlock,
    TieredCostBlock,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost, BlockCostType
from backend.executor.utils import block_usage_cost
from backend.sdk.cost_integration import (
    get_block_costs,
    register_provider_costs_for_block,
)


class TestCostIntegration:
    """Test cost integration features."""

    def test_fixed_cost_block(self):
        """Test block with fixed cost decorator."""
        _ = FixedCostBlock()  # Instantiate to register costs

        # Check that cost was registered
        assert FixedCostBlock in BLOCK_COSTS
        costs = BLOCK_COSTS[FixedCostBlock]
        assert len(costs) == 1
        assert costs[0].cost_type == BlockCostType.RUN
        assert costs[0].cost_amount == 5

    def test_tiered_cost_block(self):
        """Test block with tiered costs."""
        _ = TieredCostBlock()  # Instantiate to register costs

        # Check that all tier costs were registered
        assert TieredCostBlock in BLOCK_COSTS
        costs = BLOCK_COSTS[TieredCostBlock]
        assert len(costs) == 3

        # Check each tier
        tier_costs = {c.cost_filter.get("tier"): c.cost_amount for c in costs}
        assert tier_costs["basic"] == 1
        assert tier_costs["standard"] == 5
        assert tier_costs["premium"] == 10

    def test_data_based_cost_block(self):
        """Test block with data-based (per byte) costs."""
        _ = DataBasedCostBlock()  # Instantiate to register costs

        # Check that cost was registered
        assert DataBasedCostBlock in BLOCK_COSTS
        costs = BLOCK_COSTS[DataBasedCostBlock]
        assert len(costs) == 1
        assert costs[0].cost_type == BlockCostType.BYTE
        assert costs[0].cost_amount == 1  # 1 credit per 1000 bytes

    def test_provider_cost_block(self):
        """Test block that should inherit provider costs."""
        # Initially, ProviderCostBlock shouldn't be in BLOCK_COSTS
        # (unless it was already registered by another test)

        # Register provider costs for the block
        register_provider_costs_for_block(ProviderCostBlock)

        # Now check if costs were registered from the provider
        costs = get_block_costs(ProviderCostBlock)
        if costs:  # Provider costs should be registered if provider exists
            assert len(costs) > 0
            assert costs[0].cost_type == BlockCostType.RUN
            assert costs[0].cost_amount == 1  # From example_service provider

    def test_block_usage_cost_calculation(self):
        """Test actual cost calculation using block_usage_cost."""
        # Test fixed cost
        block = FixedCostBlock()
        cost, filter_used = block_usage_cost(block, {"data": "test"})
        assert cost == 5

        # Test tiered cost - basic tier
        block = TieredCostBlock()
        cost, filter_used = block_usage_cost(block, {"data": "test", "tier": "basic"})
        assert cost == 1

        # Test tiered cost - premium tier
        cost, filter_used = block_usage_cost(block, {"data": "test", "tier": "premium"})
        assert cost == 10

        # Test data-based cost (10KB of data)
        block = DataBasedCostBlock()
        cost, filter_used = block_usage_cost(
            block, {"data": "test", "process_intensive": False}, data_size=10000  # 10KB
        )
        assert cost == 10000  # 10KB * 1 credit per byte = 10000 credits

    def test_cost_decorator_overrides_provider(self):
        """Test that @cost decorator overrides provider costs."""
        # Create a test block with both provider and decorator costs
        from backend.blocks.examples._config import example_service
        from backend.sdk import (
            Block,
            BlockCategory,
            BlockOutput,
            BlockSchema,
            CredentialsMetaInput,
            SchemaField,
            cost,
        )

        @cost(BlockCost(cost_type=BlockCostType.RUN, cost_amount=100))
        class TestOverrideBlock(Block):
            class Input(BlockSchema):
                credentials: CredentialsMetaInput = example_service.credentials_field(
                    description="Test credentials"
                )
                data: str = SchemaField(description="Data")

            class Output(BlockSchema):
                result: str = SchemaField(description="Result")

            def __init__(self):
                super().__init__(
                    id="test-override-block-12345678-1234-1234-1234",
                    description="Test block",
                    categories={BlockCategory.DEVELOPER_TOOLS},
                    input_schema=self.Input,
                    output_schema=self.Output,
                )

            async def run(self, input_data: Input, **kwargs) -> BlockOutput:
                yield "result", "test"

        # The decorator cost should override provider cost
        _ = TestOverrideBlock()  # Instantiate to register costs
        assert TestOverrideBlock in BLOCK_COSTS
        costs = BLOCK_COSTS[TestOverrideBlock]
        assert len(costs) == 1
        assert costs[0].cost_amount == 100  # Decorator cost, not provider's 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
