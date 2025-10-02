"""Tests for metrics utility functions."""

from backend.util.metrics import _truncate_discord_content


class TestDiscordContentTruncation:
    """Test Discord content truncation functionality."""

    def test_no_truncation_needed(self):
        """Test that short content is not truncated."""
        content = "Short message"
        result = _truncate_discord_content(content)
        assert result == content
        assert len(result) <= 4000

    def test_truncation_at_exact_limit(self):
        """Test content at exact limit is not truncated."""
        content = "x" * 4000
        result = _truncate_discord_content(content)
        assert result == content
        assert len(result) == 4000

    def test_truncation_over_limit(self):
        """Test that long content is truncated."""
        content = "x" * 5000
        result = _truncate_discord_content(content)
        assert len(result) <= 4000
        assert "truncated" in result.lower()
        assert "5000" in result or "characters" in result  # Should show original length

    def test_truncation_preserves_beginning(self):
        """Test that truncation preserves the beginning of the content."""
        content = "IMPORTANT_START" + "x" * 5000 + "END"
        result = _truncate_discord_content(content)
        assert result.startswith("IMPORTANT_START")
        assert len(result) <= 4000

    def test_custom_max_length(self):
        """Test truncation with custom max length."""
        content = "x" * 3000
        result = _truncate_discord_content(content, max_length=2000)
        assert len(result) <= 2000
        assert "truncated" in result.lower()

    def test_truncation_with_newlines(self):
        """Test that truncation works with content containing newlines."""
        content = "Line 1\nLine 2\n" * 500  # Create long multi-line content
        result = _truncate_discord_content(content, max_length=1000)
        assert len(result) <= 1000
        assert "truncated" in result.lower()

    def test_unicode_content(self):
        """Test truncation with unicode characters."""
        content = "🚨 Alert: " + "x" * 5000
        result = _truncate_discord_content(content)
        assert len(result) <= 4000
        assert result.startswith("🚨 Alert:")

    def test_empty_content(self):
        """Test truncation with empty content."""
        content = ""
        result = _truncate_discord_content(content)
        assert result == ""

    def test_truncation_message_format(self):
        """Test that truncation message indicates how much was truncated."""
        original_length = 6000
        content = "x" * original_length
        result = _truncate_discord_content(content)

        # Should contain truncation notice
        assert "truncated" in result.lower()
        assert "characters" in result.lower()

        # Should not exceed max length
        assert len(result) <= 4000
