import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.chat.tools.binary_output_processor import (
    _decode_base64,
    process_binary_outputs,
)


@pytest.fixture
def workspace_manager():
    wm = AsyncMock()
    wm.write_file = AsyncMock(return_value=MagicMock(id="file-123"))
    return wm


class TestDecodeBase64:
    def test_raw_base64(self):
        assert _decode_base64(base64.b64encode(b"test").decode()) == b"test"

    def test_data_uri(self):
        encoded = base64.b64encode(b"test").decode()
        assert _decode_base64(f"data:image/png;base64,{encoded}") == b"test"

    def test_invalid_returns_none(self):
        assert _decode_base64("not base64!!!") is None


class TestProcessBinaryOutputs:
    @pytest.mark.asyncio
    async def test_saves_large_binary(self, workspace_manager):
        content = base64.b64encode(b"x" * 2000).decode()
        outputs = {"result": [{"png": content, "text": "ok"}]}

        result = await process_binary_outputs(outputs, workspace_manager, "Test")

        assert result["result"][0]["png"] == "workspace://file-123"
        assert result["result"][0]["text"] == "ok"

    @pytest.mark.asyncio
    async def test_skips_small_content(self, workspace_manager):
        content = base64.b64encode(b"tiny").decode()
        outputs = {"result": [{"png": content}]}

        result = await process_binary_outputs(outputs, workspace_manager, "Test")

        assert result["result"][0]["png"] == content
        workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_identical_content(self, workspace_manager):
        content = base64.b64encode(b"x" * 2000).decode()
        outputs = {"a": [{"pdf": content}], "b": [{"pdf": content}]}

        result = await process_binary_outputs(outputs, workspace_manager, "Test")

        assert result["a"][0]["pdf"] == result["b"][0]["pdf"] == "workspace://file-123"
        assert workspace_manager.write_file.call_count == 1

    @pytest.mark.asyncio
    async def test_failure_preserves_original(self, workspace_manager):
        workspace_manager.write_file.side_effect = Exception("Storage error")
        content = base64.b64encode(b"x" * 2000).decode()

        result = await process_binary_outputs(
            {"r": [{"png": content}]}, workspace_manager, "Test"
        )

        assert result["r"][0]["png"] == content

    @pytest.mark.asyncio
    async def test_handles_nested_structures(self, workspace_manager):
        content = base64.b64encode(b"x" * 2000).decode()
        outputs = {"result": [{"outer": {"inner": {"png": content}}}]}

        result = await process_binary_outputs(outputs, workspace_manager, "Test")

        assert result["result"][0]["outer"]["inner"]["png"] == "workspace://file-123"

    @pytest.mark.asyncio
    async def test_handles_lists_in_output(self, workspace_manager):
        content = base64.b64encode(b"x" * 2000).decode()
        outputs = {"result": [{"images": [{"png": content}, {"png": content}]}]}

        result = await process_binary_outputs(outputs, workspace_manager, "Test")

        assert result["result"][0]["images"][0]["png"] == "workspace://file-123"
        assert result["result"][0]["images"][1]["png"] == "workspace://file-123"
        # Deduplication should still work
        assert workspace_manager.write_file.call_count == 1
