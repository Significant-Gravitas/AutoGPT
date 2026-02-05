"""Tests for embedded binary detection in block outputs."""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from .binary_output_processor import (
    _decode_and_validate,
    _expand_to_markers,
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


def _make_pdf_base64(size: int = 2000) -> str:
    """Create a valid PDF base64 string of specified size."""
    pdf_content = b"%PDF-1.4 " + b"x" * size
    return base64.b64encode(pdf_content).decode()


def _make_png_base64(size: int = 2000) -> str:
    """Create a valid PNG base64 string of specified size."""
    png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * size
    return base64.b64encode(png_content).decode()


# =============================================================================
# Decode and Validate Tests
# =============================================================================


class TestDecodeAndValidate:
    """Tests for _decode_and_validate function."""

    def test_detects_pdf_magic_number(self):
        """Should detect valid PDF by magic number."""
        pdf_b64 = _make_pdf_base64()
        result = _decode_and_validate(pdf_b64)
        assert result is not None
        content, ext = result
        assert ext == "pdf"
        assert content.startswith(b"%PDF-")

    def test_detects_png_magic_number(self):
        """Should detect valid PNG by magic number."""
        png_b64 = _make_png_base64()
        result = _decode_and_validate(png_b64)
        assert result is not None
        content, ext = result
        assert ext == "png"

    def test_detects_jpeg_magic_number(self):
        """Should detect valid JPEG by magic number."""
        jpeg_content = b"\xff\xd8\xff\xe0" + b"\x00" * 2000
        jpeg_b64 = base64.b64encode(jpeg_content).decode()
        result = _decode_and_validate(jpeg_b64)
        assert result is not None
        _, ext = result
        assert ext == "jpg"

    def test_detects_gif_magic_number(self):
        """Should detect valid GIF by magic number."""
        gif_content = b"GIF89a" + b"\x00" * 2000
        gif_b64 = base64.b64encode(gif_content).decode()
        result = _decode_and_validate(gif_b64)
        assert result is not None
        _, ext = result
        assert ext == "gif"

    def test_detects_webp_magic_number(self):
        """Should detect valid WebP by magic number."""
        webp_content = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 2000
        webp_b64 = base64.b64encode(webp_content).decode()
        result = _decode_and_validate(webp_b64)
        assert result is not None
        _, ext = result
        assert ext == "webp"

    def test_rejects_small_content(self):
        """Should reject content smaller than threshold."""
        small_pdf = b"%PDF-1.4 small"
        small_b64 = base64.b64encode(small_pdf).decode()
        result = _decode_and_validate(small_b64)
        assert result is None

    def test_rejects_no_magic_number(self):
        """Should reject content without recognized magic number."""
        random_content = b"This is just random text" * 100
        random_b64 = base64.b64encode(random_content).decode()
        result = _decode_and_validate(random_b64)
        assert result is None

    def test_rejects_invalid_base64(self):
        """Should reject invalid base64."""
        result = _decode_and_validate("not-valid-base64!!!")
        assert result is None

    def test_rejects_riff_without_webp(self):
        """Should reject RIFF files that aren't WebP (e.g., WAV)."""
        wav_content = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 2000
        wav_b64 = base64.b64encode(wav_content).decode()
        result = _decode_and_validate(wav_b64)
        assert result is None


# =============================================================================
# Marker Expansion Tests
# =============================================================================


class TestExpandToMarkers:
    """Tests for _expand_to_markers function."""

    def test_expands_base64_start_end_markers(self):
        """Should expand to include ---BASE64_START--- and ---BASE64_END---."""
        text = "prefix\n---BASE64_START---\nABCDEF\n---BASE64_END---\nsuffix"
        # Base64 is at position 27-33
        start, end = _expand_to_markers(text, 27, 33)
        assert text[start:end] == "---BASE64_START---\nABCDEF\n---BASE64_END---"

    def test_expands_bracket_markers(self):
        """Should expand to include [BASE64] and [/BASE64] markers."""
        text = "prefix[BASE64]ABCDEF[/BASE64]suffix"
        # Base64 is at position 14-20
        start, end = _expand_to_markers(text, 14, 20)
        assert text[start:end] == "[BASE64]ABCDEF[/BASE64]"

    def test_no_expansion_without_markers(self):
        """Should not expand if no markers present."""
        text = "prefix ABCDEF suffix"
        start, end = _expand_to_markers(text, 7, 13)
        assert start == 7
        assert end == 13


# =============================================================================
# Process Binary Outputs Tests
# =============================================================================


class TestProcessBinaryOutputs:
    """Tests for process_binary_outputs function."""

    @pytest.mark.asyncio
    async def test_detects_embedded_pdf_in_stdout_logs(self, mock_workspace_manager):
        """Should detect and replace embedded PDF in stdout_logs."""
        pdf_b64 = _make_pdf_base64()
        stdout = f"PDF generated!\n---BASE64_START---\n{pdf_b64}\n---BASE64_END---\n"

        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "ExecuteCodeBlock"
        )

        # Should contain workspace reference, not base64
        assert "workspace://" in result["stdout_logs"][0]
        assert pdf_b64 not in result["stdout_logs"][0]
        assert "PDF generated!" in result["stdout_logs"][0]
        mock_workspace_manager.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_embedded_png_without_markers(self, mock_workspace_manager):
        """Should detect embedded PNG even without markers."""
        png_b64 = _make_png_base64()
        stdout = f"Image created: {png_b64} done"

        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "ExecuteCodeBlock"
        )

        assert "workspace://" in result["stdout_logs"][0]
        assert "Image created:" in result["stdout_logs"][0]
        assert "done" in result["stdout_logs"][0]

    @pytest.mark.asyncio
    async def test_preserves_small_strings(self, mock_workspace_manager):
        """Should not process small strings."""
        outputs = {"stdout_logs": ["small output"]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["stdout_logs"][0] == "small output"
        mock_workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_non_binary_large_strings(self, mock_workspace_manager):
        """Should preserve large strings that don't contain valid binary."""
        large_text = "A" * 5000  # Large but not base64

        outputs = {"stdout_logs": [large_text]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert result["stdout_logs"][0] == large_text
        mock_workspace_manager.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplicates_identical_content(self, mock_workspace_manager):
        """Should save identical content only once."""
        pdf_b64 = _make_pdf_base64()
        stdout1 = f"First: {pdf_b64}"
        stdout2 = f"Second: {pdf_b64}"

        outputs = {"stdout_logs": [stdout1, stdout2]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Both should have references
        assert "workspace://" in result["stdout_logs"][0]
        assert "workspace://" in result["stdout_logs"][1]
        # But only one write
        assert mock_workspace_manager.write_file.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_multiple_binaries_in_one_string(
        self, mock_workspace_manager
    ):
        """Should handle multiple embedded binaries in a single string."""
        pdf_b64 = _make_pdf_base64()
        png_b64 = _make_png_base64()
        stdout = f"PDF: {pdf_b64}\nPNG: {png_b64}"

        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Should have two workspace references
        assert result["stdout_logs"][0].count("workspace://") == 2
        assert mock_workspace_manager.write_file.call_count == 2

    @pytest.mark.asyncio
    async def test_processes_nested_structures(self, mock_workspace_manager):
        """Should recursively process nested dicts and lists."""
        pdf_b64 = _make_pdf_base64()

        outputs = {"result": [{"nested": {"deep": f"data: {pdf_b64}"}}]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert "workspace://" in result["result"][0]["nested"]["deep"]

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_save_failure(self, mock_workspace_manager):
        """Should preserve original on save failure."""
        mock_workspace_manager.write_file = AsyncMock(
            side_effect=Exception("Storage error")
        )

        pdf_b64 = _make_pdf_base64()
        stdout = f"PDF: {pdf_b64}"

        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Should keep original since save failed
        assert pdf_b64 in result["stdout_logs"][0]
