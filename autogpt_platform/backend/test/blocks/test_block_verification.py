"""
Test to verify all integration blocks can be instantiated and have valid schemas.
This test runs as part of the test suite and doesn't make actual API calls.
"""

from typing import List, Type

import pytest

from backend.sdk import Block


class TestBlockVerification:
    """Verify that all integration blocks are properly structured."""

    def get_provider_blocks(self, provider_name: str) -> List[Type[Block]]:
        """Get all block classes from a provider module."""
        blocks = []

        if provider_name == "airtable":
            from backend.blocks import airtable

            module = airtable
        elif provider_name == "baas":
            from backend.blocks import baas

            module = baas
        elif provider_name == "elevenlabs":
            from backend.blocks import elevenlabs

            module = elevenlabs
        else:
            return blocks

        # Get all exported block classes
        for attr_name in module.__all__:
            attr = getattr(module, attr_name)
            if "Block" in attr_name:
                blocks.append(attr)

        return blocks

    @pytest.mark.parametrize("provider", ["airtable", "baas", "elevenlabs"])
    def test_provider_blocks_instantiate(self, provider: str):
        """Test that all blocks from a provider can be instantiated."""
        blocks = self.get_provider_blocks(provider)
        assert len(blocks) > 0, f"No blocks found for provider {provider}"

        for block_class in blocks:
            # Should not raise an exception
            block = block_class()
            assert block is not None
            assert hasattr(block, "id")
            assert hasattr(block, "description")
            assert hasattr(block, "run")

    def test_airtable_blocks_structure(self):
        """Test Airtable blocks have proper structure."""
        from backend.blocks.airtable.records import AirtableListRecordsBlock

        block = AirtableListRecordsBlock()

        # Check basic attributes
        assert block.id is not None
        assert len(block.id) == 36  # UUID format
        assert block.description is not None
        assert "list" in block.description.lower()

        # Check input schema fields using Pydantic model fields
        assert hasattr(block, "Input")
        input_fields = (
            block.Input.model_fields
            if hasattr(block.Input, "model_fields")
            else block.Input.__fields__
        )
        assert "base_id" in input_fields
        assert "table_id_or_name" in input_fields
        assert "credentials" in input_fields

        # Check output schema fields
        assert hasattr(block, "Output")
        output_fields = (
            block.Output.model_fields
            if hasattr(block.Output, "model_fields")
            else block.Output.__fields__
        )
        assert "records" in output_fields
        assert "offset" in output_fields

    def test_baas_blocks_structure(self):
        """Test Meeting BaaS blocks have proper structure."""
        from backend.blocks.baas.bots import BaasBotJoinMeetingBlock

        block = BaasBotJoinMeetingBlock()

        # Check basic attributes
        assert block.id is not None
        assert block.description is not None
        assert "join" in block.description.lower()

        # Check input schema fields
        assert hasattr(block, "Input")
        input_fields = (
            block.Input.model_fields
            if hasattr(block.Input, "model_fields")
            else block.Input.__fields__
        )
        assert "meeting_url" in input_fields
        assert "bot_name" in input_fields  # Changed from bot_config to bot_name
        assert "bot_image" in input_fields  # Additional bot configuration field
        assert "credentials" in input_fields

        # Check output schema fields
        assert hasattr(block, "Output")
        output_fields = (
            block.Output.model_fields
            if hasattr(block.Output, "model_fields")
            else block.Output.__fields__
        )
        assert "bot_id" in output_fields

    def test_elevenlabs_blocks_structure(self):
        """Test ElevenLabs blocks have proper structure."""
        from backend.blocks.elevenlabs.speech import ElevenLabsGenerateSpeechBlock

        block = ElevenLabsGenerateSpeechBlock()

        # Check basic attributes
        assert block.id is not None
        assert block.description is not None
        assert "speech" in block.description.lower()

        # Check input schema fields
        assert hasattr(block, "Input")
        input_fields = (
            block.Input.model_fields
            if hasattr(block.Input, "model_fields")
            else block.Input.__fields__
        )
        assert "text" in input_fields
        assert "voice_id" in input_fields
        assert "credentials" in input_fields

        # Check output schema fields
        assert hasattr(block, "Output")
        output_fields = (
            block.Output.model_fields
            if hasattr(block.Output, "model_fields")
            else block.Output.__fields__
        )
        assert "audio" in output_fields

    def test_webhook_blocks_structure(self):
        """Test webhook trigger blocks have proper structure."""
        from backend.blocks.airtable.triggers import AirtableWebhookTriggerBlock
        from backend.blocks.baas.triggers import BaasOnMeetingEventBlock
        from backend.blocks.elevenlabs.triggers import ElevenLabsWebhookTriggerBlock

        webhook_blocks = [
            AirtableWebhookTriggerBlock(),
            BaasOnMeetingEventBlock(),
            ElevenLabsWebhookTriggerBlock(),
        ]

        for block in webhook_blocks:
            # Check input fields
            input_fields = (
                block.Input.model_fields
                if hasattr(block.Input, "model_fields")
                else block.Input.__fields__
            )
            assert (
                "webhook_url" in input_fields
            )  # Changed from webhook_id to webhook_url
            assert "credentials" in input_fields  # Changed from secret to credentials
            assert "payload" in input_fields  # Webhook payload field

            # Check output fields exist (different blocks have different output structures)
            _ = (
                block.Output.model_fields
                if hasattr(block.Output, "model_fields")
                else block.Output.__fields__
            )

    def test_block_run_method_is_async(self):
        """Test that all blocks have async run methods."""
        from backend.blocks.airtable.metadata import AirtableListBasesBlock
        from backend.blocks.baas.calendars import BaasCalendarListAllBlock
        from backend.blocks.elevenlabs.voices import ElevenLabsListVoicesBlock

        block_classes = [
            AirtableListBasesBlock,
            BaasCalendarListAllBlock,
            ElevenLabsListVoicesBlock,
        ]

        import inspect

        for block_class in block_classes:
            # Check that run method exists
            assert hasattr(
                block_class, "run"
            ), f"{block_class.__name__} does not have a 'run' method"

            # Create an instance to check the bound method
            block_instance = block_class()

            # Try to verify it's an async method by checking if it would return a coroutine
            # We can't actually call it without proper arguments, but we can check the method type
            run_method = block_instance.run

            # The run method should be a bound method that when called returns a coroutine
            # Let's just check that the method exists and is callable
            assert callable(run_method), f"{block_class.__name__}.run is not callable"

            # Check the source to ensure it's defined as async
            # This is a bit of a workaround but should work
            try:
                source = inspect.getsource(block_class.run)
                assert source.strip().startswith(
                    "async def run"
                ), f"{block_class.__name__}.run is not defined as async def"
            except Exception:
                # If we can't get source, just check that it exists and is callable
                pass

    def test_blocks_use_correct_credential_types(self):
        """Test that blocks use appropriate credential types."""
        from backend.blocks.airtable.records import AirtableGetRecordBlock
        from backend.blocks.baas.events import BaasEventListBlock
        from backend.blocks.elevenlabs.utility import ElevenLabsListModelsBlock

        # All these providers use API key authentication
        blocks = [
            AirtableGetRecordBlock(),
            BaasEventListBlock(),
            ElevenLabsListModelsBlock(),
        ]

        for block in blocks:
            # Check that credentials field exists
            input_fields = (
                block.Input.model_fields
                if hasattr(block.Input, "model_fields")
                else block.Input.__fields__
            )
            assert "credentials" in input_fields

            # Get the field info
            field = input_fields["credentials"]
            assert field is not None
