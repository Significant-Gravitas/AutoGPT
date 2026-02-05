"""Tests for content-based binary output detection and saving."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from .binary_output_processor import (
    _detect_data_uri,
    _detect_raw_base64,
    _mimetype_to_ext,
    process_binary_outputs,
)


@pytest.fixture
def mock_workspace_manager():
    """Create a mock workspace manager that returns predictable file IDs."""
    wm = MagicMock()

    async def mock_write_file(content, filename):
        file = MagicMock()
        file.id = f"file-{filename[:10]}"
        return file

    wm.write_file = AsyncMock(side_effect=mock_write_file)
    return wm


# =============================================================================
# Data URI Detection Tests
# =============================================================================


class TestDetectDataUri:
    """Tests for _detect_data_uri function."""

    def test_detects_png_data_uri(self):
        """Should detect valid PNG data URI."""
        # Minimal valid PNG (1x1 transparent)
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        data_uri = f"data:image/png;base64,{png_b64}"

        result = _detect_data_uri(data_uri)

        assert result is not None
        content, ext = result
        assert ext == "png"
        assert content.startswith(b"\x89PNG")

    def test_detects_pdf_data_uri(self):
        """Should detect valid PDF data URI."""
        pdf_content = b"%PDF-1.4 test content"
        pdf_b64 = base64.b64encode(pdf_content).decode()
        data_uri = f"data:application/pdf;base64,{pdf_b64}"

        result = _detect_data_uri(data_uri)

        assert result is not None
        content, ext = result
        assert ext == "pdf"
        assert content == pdf_content

    def test_rejects_text_plain_mimetype(self):
        """Should reject text/plain mimetype (not in whitelist)."""
        text_b64 = base64.b64encode(b"Hello World").decode()
        data_uri = f"data:text/plain;base64,{text_b64}"

        result = _detect_data_uri(data_uri)

        assert result is None

    def test_rejects_non_data_uri_string(self):
        """Should return None for non-data-URI strings."""
        result = _detect_data_uri("https://example.com/image.png")
        assert result is None

    def test_rejects_invalid_base64_in_data_uri(self):
        """Should return None for data URI with invalid base64."""
        data_uri = "data:image/png;base64,not-valid-base64!!!"
        result = _detect_data_uri(data_uri)
        assert result is None

    def test_handles_jpeg_mimetype(self):
        """Should handle image/jpeg mimetype."""
        jpeg_content = b"\xff\xd8\xff\xe0test"
        jpeg_b64 = base64.b64encode(jpeg_content).decode()
        data_uri = f"data:image/jpeg;base64,{jpeg_b64}"

        result = _detect_data_uri(data_uri)

        assert result is not None
        _, ext = result
        assert ext == "jpg"


# =============================================================================
# Raw Base64 Detection Tests
# =============================================================================


class TestDetectRawBase64:
    """Tests for _detect_raw_base64 function."""

    def test_detects_png_magic_number(self):
        """Should detect raw base64 PNG by magic number."""
        # Minimal valid PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        result = _detect_raw_base64(png_b64)

        assert result is not None
        content, ext = result
        assert ext == "png"
        assert content.startswith(b"\x89PNG")

    def test_detects_jpeg_magic_number(self):
        """Should detect raw base64 JPEG by magic number."""
        jpeg_content = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        jpeg_b64 = base64.b64encode(jpeg_content).decode()

        result = _detect_raw_base64(jpeg_b64)

        assert result is not None
        _, ext = result
        assert ext == "jpg"

    def test_detects_pdf_magic_number(self):
        """Should detect raw base64 PDF by magic number."""
        pdf_content = b"%PDF-1.4 " + b"x" * 100
        pdf_b64 = base64.b64encode(pdf_content).decode()

        result = _detect_raw_base64(pdf_b64)

        assert result is not None
        _, ext = result
        assert ext == "pdf"

    def test_detects_gif87a_magic_number(self):
        """Should detect GIF87a magic number."""
        gif_content = b"GIF87a" + b"\x00" * 100
        gif_b64 = base64.b64encode(gif_content).decode()

        result = _detect_raw_base64(gif_b64)

        assert result is not None
        _, ext = result
        assert ext == "gif"

    def test_detects_gif89a_magic_number(self):
        """Should detect GIF89a magic number."""
        gif_content = b"GIF89a" + b"\x00" * 100
        gif_b64 = base64.b64encode(gif_content).decode()

        result = _detect_raw_base64(gif_b64)

        assert result is not None
        _, ext = result
        assert ext == "gif"

    def test_detects_webp_magic_number(self):
        """Should detect WebP (RIFF + WEBP at offset 8)."""
        # WebP header: RIFF + size (4 bytes) + WEBP
        webp_content = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
        webp_b64 = base64.b64encode(webp_content).decode()

        result = _detect_raw_base64(webp_b64)

        assert result is not None
        _, ext = result
        assert ext == "webp"

    def test_rejects_riff_without_webp(self):
        """Should reject RIFF files that aren't WebP (e.g., WAV)."""
        wav_content = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100
        wav_b64 = base64.b64encode(wav_content).decode()

        result = _detect_raw_base64(wav_b64)

        assert result is None

    def test_rejects_non_base64_string(self):
        """Should reject strings that don't look like base64."""
        result = _detect_raw_base64("Hello, this is regular text with spaces!")
        assert result is None

    def test_rejects_base64_without_magic_number(self):
        """Should reject valid base64 that doesn't have a known magic number."""
        random_content = b"This is just random text, not a binary file"
        random_b64 = base64.b64encode(random_content).decode()

        result = _detect_raw_base64(random_b64)

        assert result is None

    def test_rejects_invalid_base64(self):
        """Should return None for invalid base64."""
        result = _detect_raw_base64("not-valid-base64!!!")
        assert result is None

    def test_detects_base64_with_line_breaks(self):
        """Should detect raw base64 with RFC 2045 line breaks."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        png_b64 = base64.b64encode(png_content).decode()
        # Simulate RFC 2045 line wrapping at 76 chars
        wrapped = png_b64[:76] + "\n" + png_b64[76:]

        result = _detect_raw_base64(wrapped)

        assert result is not None
        content, ext = result
        assert ext == "png"
        assert content == png_content


# =============================================================================
# Process Binary Outputs Tests
# =============================================================================


class TestProcessBinaryOutputs:
    """Tests for process_binary_outputs function."""

    @pytest.mark.asyncio
    async def test_saves_large_png_and_returns_reference(self, mock_workspace_manager):
        """Should save PNG > 1KB and return workspace reference."""
        # Create PNG > 1KB
        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()

        outputs = {"result": [png_b64]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0].startswith("workspace://")
        mock_workspace_manager.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserves_small_content(self, mock_workspace_manager):
        """Should not process strings smaller than threshold."""
        small_content = "small"

        outputs = {"result": [small_content]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0] == small_content
        mock_workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_identical_content(self, mock_workspace_manager):
        """Should save identical content only once."""
        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()

        outputs = {"result": [png_b64, png_b64]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Both should have references
        assert result["result"][0].startswith("workspace://")
        assert result["result"][1].startswith("workspace://")
        # But only one write should have happened
        assert mock_workspace_manager.write_file.call_count == 1
        # And they should be the same reference
        assert result["result"][0] == result["result"][1]

    @pytest.mark.asyncio
    async def test_processes_nested_dict(self, mock_workspace_manager):
        """Should recursively process nested dictionaries."""
        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()

        outputs = {"result": [{"nested": {"deep": png_b64}}]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0]["nested"]["deep"].startswith("workspace://")

    @pytest.mark.asyncio
    async def test_processes_nested_list(self, mock_workspace_manager):
        """Should recursively process nested lists."""
        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()

        outputs = {"result": [[png_b64]]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0][0].startswith("workspace://")

    @pytest.mark.asyncio
    async def test_handles_data_uri_format(self, mock_workspace_manager):
        """Should handle data URI format."""
        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()
        data_uri = f"data:image/png;base64,{png_b64}"

        outputs = {"result": [data_uri]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0].startswith("workspace://")

    @pytest.mark.asyncio
    async def test_preserves_non_binary_large_strings(self, mock_workspace_manager):
        """Should preserve large strings that aren't binary."""
        large_text = "A" * 2000  # Large but not base64 or binary

        outputs = {"result": [large_text]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["result"][0] == large_text
        mock_workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_save_failure(self, mock_workspace_manager):
        """Should preserve original value if save fails."""
        mock_workspace_manager.write_file = AsyncMock(
            side_effect=Exception("Storage error")
        )

        png_header = b"\x89PNG\r\n\x1a\n"
        png_content = png_header + b"\x00" * 2000
        png_b64 = base64.b64encode(png_content).decode()

        outputs = {"result": [png_b64]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Should return original value on failure
        assert result["result"][0] == png_b64

    @pytest.mark.asyncio
    async def test_handles_stdout_logs_field(self, mock_workspace_manager):
        """Should detect binary in stdout_logs (the actual failing case)."""
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()

        outputs = {"stdout_logs": [pdf_b64]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "ExecuteCodeBlock"
        )

        assert result["stdout_logs"][0].startswith("workspace://")


# =============================================================================
# Mimetype to Extension Tests
# =============================================================================


class TestMimetypeToExt:
    """Tests for _mimetype_to_ext function."""

    def test_png_mapping(self):
        assert _mimetype_to_ext("image/png") == "png"

    def test_jpeg_mapping(self):
        assert _mimetype_to_ext("image/jpeg") == "jpg"

    def test_nonstandard_jpg_mapping(self):
        assert _mimetype_to_ext("image/jpg") == "jpg"

    def test_gif_mapping(self):
        assert _mimetype_to_ext("image/gif") == "gif"

    def test_webp_mapping(self):
        assert _mimetype_to_ext("image/webp") == "webp"

    def test_svg_mapping(self):
        assert _mimetype_to_ext("image/svg+xml") == "svg"

    def test_pdf_mapping(self):
        assert _mimetype_to_ext("application/pdf") == "pdf"

    def test_octet_stream_mapping(self):
        assert _mimetype_to_ext("application/octet-stream") == "bin"

    def test_unknown_mimetype(self):
        assert _mimetype_to_ext("application/unknown") == "bin"
