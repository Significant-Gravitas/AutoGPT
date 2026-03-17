from backend.util.exceptions import (
    BlockError,
    BlockExecutionError,
    BlockInputError,
    BlockOutputError,
    BlockUnknownError,
)


class TestBlockError:
    """Tests for BlockError and its subclasses."""

    def test_block_error_message_format(self):
        """Test that BlockError formats the message correctly."""
        error = BlockError(
            message="Test error", block_name="TestBlock", block_id="test-123"
        )
        assert (
            str(error)
            == "raised by TestBlock with message: Test error. block_id: test-123"
        )

    def test_block_input_error_inherits_format(self):
        """Test that BlockInputError uses parent's message format."""
        error = BlockInputError(
            message="Invalid input", block_name="TestBlock", block_id="test-123"
        )
        assert "raised by TestBlock with message: Invalid input" in str(error)

    def test_block_output_error_inherits_format(self):
        """Test that BlockOutputError uses parent's message format."""
        error = BlockOutputError(
            message="Invalid output", block_name="TestBlock", block_id="test-123"
        )
        assert "raised by TestBlock with message: Invalid output" in str(error)


class TestBlockExecutionErrorNoneHandling:
    """Tests for BlockExecutionError handling of None messages."""

    def test_execution_error_with_none_message(self):
        """Test that None message is replaced with descriptive text."""
        error = BlockExecutionError(
            message=None, block_name="TestBlock", block_id="test-123"
        )
        assert "Output error was None" in str(error)
        assert "raised by TestBlock with message: Output error was None" in str(error)

    def test_execution_error_with_valid_message(self):
        """Test that valid messages are preserved."""
        error = BlockExecutionError(
            message="Actual error", block_name="TestBlock", block_id="test-123"
        )
        assert "Actual error" in str(error)
        assert "Output error was None" not in str(error)

    def test_execution_error_with_empty_string(self):
        """Test that empty string message is NOT replaced (only None is)."""
        error = BlockExecutionError(
            message="", block_name="TestBlock", block_id="test-123"
        )
        # Empty string is falsy but not None, so it's preserved
        assert "raised by TestBlock with message: . block_id:" in str(error)


class TestBlockUnknownErrorNoneHandling:
    """Tests for BlockUnknownError handling of None/empty messages."""

    def test_unknown_error_with_none_message(self):
        """Test that None message is replaced with descriptive text."""
        error = BlockUnknownError(
            message=None, block_name="TestBlock", block_id="test-123"
        )
        assert "Unknown error occurred" in str(error)

    def test_unknown_error_with_empty_string(self):
        """Test that empty string is replaced with descriptive text."""
        error = BlockUnknownError(
            message="", block_name="TestBlock", block_id="test-123"
        )
        assert "Unknown error occurred" in str(error)

    def test_unknown_error_with_valid_message(self):
        """Test that valid messages are preserved."""
        error = BlockUnknownError(
            message="Something went wrong", block_name="TestBlock", block_id="test-123"
        )
        assert "Something went wrong" in str(error)
        assert "Unknown error occurred" not in str(error)


class TestBlockErrorInheritance:
    """Tests for proper exception inheritance."""

    def test_block_execution_error_is_value_error(self):
        """Test that BlockExecutionError is a ValueError."""
        error = BlockExecutionError(
            message="test", block_name="TestBlock", block_id="test-123"
        )
        assert isinstance(error, ValueError)
        assert isinstance(error, BlockError)

    def test_block_input_error_is_value_error(self):
        """Test that BlockInputError is a ValueError."""
        error = BlockInputError(
            message="test", block_name="TestBlock", block_id="test-123"
        )
        assert isinstance(error, ValueError)
        assert isinstance(error, BlockError)

    def test_block_output_error_is_value_error(self):
        """Test that BlockOutputError is a ValueError."""
        error = BlockOutputError(
            message="test", block_name="TestBlock", block_id="test-123"
        )
        assert isinstance(error, ValueError)
        assert isinstance(error, BlockError)

    def test_block_unknown_error_is_not_value_error(self):
        """Test that BlockUnknownError is NOT a ValueError."""
        error = BlockUnknownError(
            message="test", block_name="TestBlock", block_id="test-123"
        )
        assert not isinstance(error, ValueError)
        assert isinstance(error, BlockError)
