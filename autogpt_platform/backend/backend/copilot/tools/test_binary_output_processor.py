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

    def test_handles_line_wrapped_base64(self):
        """Should handle RFC 2045 line-wrapped base64."""
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()
        # Simulate line wrapping at 76 chars
        wrapped = "\n".join(pdf_b64[i : i + 76] for i in range(0, len(pdf_b64), 76))
        result = _decode_and_validate(wrapped)
        assert result is not None
        content, ext = result
        assert ext == "pdf"
        assert content == pdf_content


# =============================================================================
# Marker Expansion Tests
# =============================================================================


class TestExpandToMarkers:
    """Tests for _expand_to_markers function."""

    def test_expands_base64_start_end_markers(self):
        """Should expand to include ---BASE64_START--- and ---BASE64_END---."""
        text = "prefix\n---BASE64_START---\nABCDEF\n---BASE64_END---\nsuffix"
        # Base64 "ABCDEF" is at position 26-32
        start, end = _expand_to_markers(text, 26, 32)
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
        large_text = "A" * 5000  # Large string - decodes to nulls, no magic number

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


# =============================================================================
# Offset Loop Tests (handling marker bleed-in)
# =============================================================================


class TestOffsetLoopHandling:
    """Tests for the offset-aligned decoding that handles marker bleed-in."""

    def test_handles_4char_aligned_prefix(self):
        """Should detect base64 when a 4-char aligned prefix bleeds into match.

        When 'TEST' (4 chars, aligned) bleeds in, offset 4 finds valid base64.
        """
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()
        # 4-char prefix (aligned)
        with_prefix = f"TEST{pdf_b64}"

        result = _decode_and_validate(with_prefix)
        assert result is not None
        content, ext = result
        assert ext == "pdf"
        assert content == pdf_content

    def test_handles_8char_aligned_prefix(self):
        """Should detect base64 when an 8-char prefix bleeds into match."""
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()
        # 8-char prefix (aligned)
        with_prefix = f"TESTTEST{pdf_b64}"

        result = _decode_and_validate(with_prefix)
        assert result is not None
        content, ext = result
        assert ext == "pdf"

    def test_misaligned_prefix_not_detected(self):
        """Misaligned prefixes (length not divisible by 4) are NOT recoverable.

        The loop tries 4-byte aligned offsets: 0, 4, 8, 12...
        A 5-char 'START' prefix means no aligned offset lands cleanly at the
        start of the base64 payload — each offset decodes a corrupted sequence
        that won't match any magic number. This is a known limitation.

        Only markers whose length is divisible by 4 (like PDF_BASE64_START=16)
        can be skipped by the offset loop.
        """
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()
        with_prefix = f"START{pdf_b64}"  # 5-char prefix — NOT 4-aligned

        result = _decode_and_validate(with_prefix)
        # Misaligned prefix: no aligned offset can recover the PDF magic bytes
        assert result is None

    def test_handles_pdf_base64_start_marker_bleed(self):
        """Should handle PDF_BASE64_START marker bleeding into regex match.

        This is the real-world case: regex matches 'STARTJVBERi0...' because
        'START' chars are in the base64 alphabet. Offset loop skips past it.
        PDF_BASE64_START is 16 chars (4-aligned), so offset 16 finds valid base64.
        """
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()
        # Simulate regex capturing 'PDF_BASE64_START' + base64 together
        # This happens when there's no delimiter between marker and content
        with_full_marker = f"PDF_BASE64_START{pdf_b64}"

        result = _decode_and_validate(with_full_marker)
        assert result is not None
        _, ext = result
        assert ext == "pdf"

    def test_clean_base64_works_at_offset_zero(self):
        """Should detect clean base64 at offset 0 without issues."""
        pdf_content = b"%PDF-1.4 " + b"x" * 2000
        pdf_b64 = base64.b64encode(pdf_content).decode()

        result = _decode_and_validate(pdf_b64)
        assert result is not None
        content, ext = result
        assert ext == "pdf"
        assert content == pdf_content


# =============================================================================
# PDF Marker Tests
# =============================================================================


class TestPdfMarkerExpansion:
    """Tests for PDF_BASE64_START/END marker handling."""

    def test_expands_pdf_base64_start_marker(self):
        """Should expand to include PDF_BASE64_START marker."""
        text = "prefixPDF_BASE64_STARTABCDEF"
        # Base64 'ABCDEF' is at position 22-28
        start, end = _expand_to_markers(text, 22, 28)
        assert text[start:end] == "PDF_BASE64_STARTABCDEF"

    def test_expands_pdf_base64_end_marker(self):
        """Should expand to include PDF_BASE64_END marker."""
        text = "ABCDEFPDF_BASE64_ENDsuffix"
        # Base64 'ABCDEF' is at position 0-6
        start, end = _expand_to_markers(text, 0, 6)
        assert text[start:end] == "ABCDEFPDF_BASE64_END"

    def test_expands_both_pdf_markers(self):
        """Should expand to include both PDF_BASE64_START and END."""
        text = "xPDF_BASE64_STARTABCDEFPDF_BASE64_ENDy"
        # Base64 'ABCDEF' is at position 17-23
        start, end = _expand_to_markers(text, 17, 23)
        assert text[start:end] == "PDF_BASE64_STARTABCDEFPDF_BASE64_END"

    def test_partial_marker_not_expanded(self):
        """Should not expand if only partial marker present."""
        text = "BASE64_STARTABCDEF"  # Missing 'PDF_' prefix
        start, end = _expand_to_markers(text, 12, 18)
        # Should not expand since it's not the full marker
        assert start == 12
        assert end == 18

    @pytest.mark.asyncio
    async def test_full_pipeline_with_base64_markers(self, mock_workspace_manager):
        """Test full pipeline with ---BASE64_START/END--- markers (end-to-end).

        Note: PDF_BASE64_START/END markers contain underscores which break the
        regex match before 'START', leaving a 5-char non-4-aligned bleed that
        cannot be recovered by the offset loop. Their unit-level behavior
        (text expansion when the regex matches correctly) is tested in
        test_expands_pdf_base64_start_marker above.
        """
        pdf_b64 = _make_pdf_base64()
        stdout = f"Output: ---BASE64_START---\n{pdf_b64}\n---BASE64_END--- done"

        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        # Should have workspace reference
        assert "workspace://" in result["stdout_logs"][0]
        # Markers should be consumed along with base64
        assert "---BASE64_START---" not in result["stdout_logs"][0]
        assert "---BASE64_END---" not in result["stdout_logs"][0]
        # Surrounding text preserved
        assert "Output:" in result["stdout_logs"][0]
        assert "done" in result["stdout_logs"][0]


# =============================================================================
# Workspace Write Tests
# =============================================================================


class TestWorkspaceWrites:
    """Tests for workspace write integration."""

    @pytest.mark.asyncio
    async def test_writes_binary_to_workspace_manager(self, mock_workspace_manager):
        """Should delegate saving to WorkspaceManager."""
        pdf_b64 = _make_pdf_base64()
        stdout = f"PDF: {pdf_b64}"
        outputs = {"stdout_logs": [stdout]}

        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        mock_workspace_manager.write_file.assert_called_once()
        assert "workspace://" in result["stdout_logs"][0]

    @pytest.mark.asyncio
    async def test_workspace_write_failure_preserves_original(
        self, mock_workspace_manager
    ):
        """Should preserve original if workspace write fails."""
        pdf_b64 = _make_pdf_base64()
        stdout = f"PDF: {pdf_b64}"
        outputs = {"stdout_logs": [stdout]}

        mock_workspace_manager.write_file = AsyncMock(
            side_effect=Exception("Storage error")
        )
        result = await process_binary_outputs(
            outputs, mock_workspace_manager, "TestBlock"
        )

        assert pdf_b64 in result["stdout_logs"][0]
        mock_workspace_manager.write_file.assert_called_once()
