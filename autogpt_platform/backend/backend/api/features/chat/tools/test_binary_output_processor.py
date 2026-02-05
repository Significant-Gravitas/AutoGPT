"""Unit tests for binary_output_processor module."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.chat.tools.binary_output_processor import (
    _decode_base64,
    process_binary_outputs,
)


@pytest.fixture
def workspace_manager():
    """Create a mock WorkspaceManager."""
    mock = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file-123"
    mock.write_file = AsyncMock(return_value=mock_file)
    return mock


class TestDecodeBase64:
    """Tests for _decode_base64 function."""

    def test_raw_base64(self):
        """Decode raw base64 string."""
        encoded = base64.b64encode(b"test content").decode()
        result = _decode_base64(encoded)
        assert result == b"test content"

    def test_data_uri(self):
        """Decode base64 from data URI format."""
        content = b"test content"
        encoded = base64.b64encode(content).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        result = _decode_base64(data_uri)
        assert result == content

    def test_invalid_base64(self):
        """Return None for invalid base64."""
        result = _decode_base64("not valid base64!!!")
        assert result is None

    def test_missing_padding(self):
        """Handle base64 with missing padding."""
        # base64.b64encode(b"test") = "dGVzdA=="
        # Remove padding
        result = _decode_base64("dGVzdA")
        assert result == b"test"

    def test_malformed_data_uri(self):
        """Return None for data URI without comma."""
        result = _decode_base64("data:image/png;base64")
        assert result is None


class TestProcessBinaryOutputs:
    """Tests for process_binary_outputs function."""

    @pytest.mark.asyncio
    async def test_saves_large_png(self, workspace_manager):
        """Large PNG content should be saved to workspace."""
        # Create content larger than SIZE_THRESHOLD (1KB)
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        outputs = {"result": [{"png": encoded}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["png"] == "workspace://file-123"
        workspace_manager.write_file.assert_called_once()
        call_kwargs = workspace_manager.write_file.call_args.kwargs
        assert call_kwargs["content"] == large_content
        assert "testblock_png_" in call_kwargs["filename"]
        assert call_kwargs["filename"].endswith(".png")

    @pytest.mark.asyncio
    async def test_preserves_small_content(self, workspace_manager):
        """Small content should be preserved as-is."""
        small_content = base64.b64encode(b"tiny").decode()
        outputs = {"result": [{"png": small_content}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["png"] == small_content
        workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_identical_content(self, workspace_manager):
        """Identical content should only be saved once."""
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        outputs = {
            "main_result": [{"png": encoded}],
            "results": [{"png": encoded}],
        }

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        # Both should have the same workspace reference
        assert result["main_result"][0]["png"] == "workspace://file-123"
        assert result["results"][0]["png"] == "workspace://file-123"
        # But write_file should only be called once
        assert workspace_manager.write_file.call_count == 1

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_save_failure(self, workspace_manager):
        """Original content should be preserved if save fails."""
        workspace_manager.write_file = AsyncMock(side_effect=Exception("Save failed"))
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        outputs = {"result": [{"png": encoded}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        # Original content should be preserved
        assert result["result"][0]["png"] == encoded

    @pytest.mark.asyncio
    async def test_handles_nested_structures(self, workspace_manager):
        """Should traverse nested dicts and lists."""
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        outputs = {
            "result": [
                {
                    "nested": {
                        "deep": {
                            "png": encoded,
                        }
                    }
                }
            ]
        }

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["nested"]["deep"]["png"] == "workspace://file-123"

    @pytest.mark.asyncio
    async def test_handles_svg_as_text(self, workspace_manager):
        """SVG should be saved as UTF-8 text, not base64 decoded."""
        svg_content = "<svg>" + "x" * 2000 + "</svg>"
        outputs = {"result": [{"svg": svg_content}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["svg"] == "workspace://file-123"
        call_kwargs = workspace_manager.write_file.call_args.kwargs
        # SVG should be UTF-8 encoded, not base64 decoded
        assert call_kwargs["content"] == svg_content.encode("utf-8")
        assert call_kwargs["filename"].endswith(".svg")

    @pytest.mark.asyncio
    async def test_ignores_unknown_fields(self, workspace_manager):
        """Fields not in SAVEABLE_FIELDS should be ignored."""
        large_content = "x" * 2000  # Large text in an unknown field
        outputs = {"result": [{"unknown_field": large_content}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["unknown_field"] == large_content
        workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_jpeg_extension(self, workspace_manager):
        """JPEG files should use .jpg extension."""
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        outputs = {"result": [{"jpeg": encoded}]}

        await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        call_kwargs = workspace_manager.write_file.call_args.kwargs
        assert call_kwargs["filename"].endswith(".jpg")

    @pytest.mark.asyncio
    async def test_handles_data_uri_in_binary_field(self, workspace_manager):
        """Data URI format in binary fields should be properly decoded."""
        large_content = b"x" * 2000
        encoded = base64.b64encode(large_content).decode()
        data_uri = f"data:image/png;base64,{encoded}"
        outputs = {"result": [{"png": data_uri}]}

        result = await process_binary_outputs(outputs, workspace_manager, "TestBlock")

        assert result["result"][0]["png"] == "workspace://file-123"
        call_kwargs = workspace_manager.write_file.call_args.kwargs
        assert call_kwargs["content"] == large_content

    @pytest.mark.asyncio
    async def test_invalid_base64_preserves_original(self, workspace_manager):
        """Invalid base64 in a binary field should preserve the original value."""
        invalid_content = "not valid base64!!!" + "x" * 2000
        outputs = {"result": [{"png": invalid_content}]}

        processed = await process_binary_outputs(
            outputs, workspace_manager, "TestBlock"
        )

        assert processed["result"][0]["png"] == invalid_content
        workspace_manager.write_file.assert_not_called()
