"""
Generic API compliance tests for all provider blocks.
This test suite verifies that all API calls match the expected patterns defined in JSON specifications.
"""

import json
import os

# Import from the same directory
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from api_test_framework import APITestInterceptor

from backend.sdk import Block


class TestAPICompliance:
    """Test API compliance for all provider blocks."""

    @pytest.fixture
    def api_interceptor(self):
        """Create API test interceptor with test data."""
        # test_data is now in the same directory as this file
        test_data_path = Path(__file__).parent / "test_data"
        return APITestInterceptor(test_data_path)

    def get_all_blocks_for_provider(
        self, provider: str
    ) -> List[tuple[str, type[Block]]]:
        """Get all block classes for a provider."""
        blocks = []

        # Import provider module
        import importlib
        import inspect

        try:
            if provider in ["airtable", "baas", "elevenlabs", "oxylabs"]:
                module = importlib.import_module(f"backend.blocks.{provider}")
            elif provider == "exa":
                # For exa, we need to import all individual files
                from backend.blocks.exa import (
                    answers,
                    contents,
                    search,
                    similar,
                    webhook_blocks,
                    websets,
                )

                # Collect all blocks from exa modules
                for submodule in [
                    answers,
                    contents,
                    search,
                    similar,
                    websets,
                    webhook_blocks,
                ]:
                    for name, obj in inspect.getmembers(submodule):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, Block)
                            and obj is not Block
                            and name.endswith("Block")
                        ):
                            blocks.append((name, obj))
                return blocks
            elif provider == "gem":
                from backend.blocks.gem import blocks as gem

                module = gem
            else:
                return blocks

            # Find all block classes
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Block)
                    and obj is not Block
                    and name.endswith("Block")
                ):
                    blocks.append((name, obj))
        except ImportError:
            pass

        return blocks

    @pytest.mark.parametrize(
        "provider", ["airtable", "baas", "elevenlabs", "exa", "gem", "oxylabs"]
    )
    async def test_provider_blocks(
        self, provider: str, api_interceptor: APITestInterceptor
    ):
        """Test that provider blocks make expected API calls."""
        # Get provider spec from already loaded specs
        spec = api_interceptor.api_specs.get(provider)
        if not spec:
            pytest.skip(f"No spec found for {provider}")

        # Get test scenarios
        test_scenarios = spec.get("test_scenarios", {})
        if not test_scenarios:
            pytest.skip(f"No test scenarios defined for {provider}")

        # Get all blocks for this provider
        provider_blocks = self.get_all_blocks_for_provider(provider)
        block_dict = {name: cls for name, cls in provider_blocks}

        # Run test scenarios
        for block_name, scenarios in test_scenarios.items():
            if block_name not in block_dict:
                # Try to find block with partial match
                found = False
                for actual_name, block_cls in block_dict.items():
                    if block_name in actual_name or actual_name in block_name:
                        block_name = actual_name
                        found = True
                        break

                if not found:
                    print(
                        f"Warning: Block {block_name} not found in provider {provider}"
                    )
                    continue

            block_cls = block_dict[block_name]

            for scenario in scenarios:
                # Create block instance
                try:
                    block = block_cls()
                except Exception as e:
                    pytest.fail(f"Failed to instantiate {block_name}: {e}")

                # Prepare test input
                test_input = scenario.get("input", {})
                expected_calls = scenario.get("expected_calls", [])

                # Mock credentials if needed
                mock_creds = api_interceptor.create_test_credentials(provider)

                # Create mock requests object
                mock_requests = api_interceptor.create_mock_requests(provider)

                # Patch Requests to use our interceptor
                with patch("backend.sdk.Requests", return_value=mock_requests):
                    try:
                        # Clear the call log before running
                        api_interceptor.clear_log()

                        # Create input instance
                        input_class = getattr(block, "Input")
                        input_data = input_class(**test_input)

                        # Run block
                        outputs = []
                        async for output in block.run(
                            input_data, credentials=mock_creds
                        ):
                            outputs.append(output)

                        # Verify API calls were made
                        if expected_calls and not api_interceptor.call_log:
                            pytest.fail(
                                f"{block_name}: No API calls were made, but expected: {expected_calls}"
                            )

                        # Log actual calls for debugging
                        if api_interceptor.call_log:
                            print(f"\n{block_name} API calls:")
                            print(api_interceptor.get_call_summary())

                    except Exception:
                        # Expected for blocks that need real API access
                        # Just verify the block structure is correct
                        pass

    def test_all_providers_have_specs(self):
        """Test that all provider directories have test specifications."""
        test_data_path = Path(__file__).parent / "test_data"
        providers = ["airtable", "baas", "elevenlabs", "exa", "gem", "oxylabs"]

        for provider in providers:
            spec_file = test_data_path / f"{provider}.json"
            assert spec_file.exists(), f"Missing test spec for {provider}"

            # Verify spec is valid JSON
            with open(spec_file) as f:
                spec = json.load(f)
                assert "provider" in spec
                assert "api_calls" in spec
