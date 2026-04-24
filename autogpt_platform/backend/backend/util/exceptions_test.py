from backend.util.exceptions import (
    BlockError,
    BlockExecutionError,
    BlockInputError,
    BlockOutputError,
    BlockUnknownError,
)


class TestBlockError:
    """Tests for BlockError and its subclasses."""

    def test_block_error_surfaces_message_unframed(self):
        """``str(exc)`` is just the message so ``yield "error", str(exc)``
        shows the actual upstream cause to the user instead of wrapping it
        in a "raised by X with message: Y" framing."""
        error = BlockError(
            message="Test error", block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == "Test error"
        assert error.block_name == "TestBlock"
        assert error.block_id == "test-123"

    def test_block_input_error_inherits_format(self):
        error = BlockInputError(
            message="Invalid input", block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == "Invalid input"
        assert error.block_name == "TestBlock"

    def test_block_output_error_inherits_format(self):
        error = BlockOutputError(
            message="Invalid output", block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == "Invalid output"


class TestBlockExecutionErrorNoneHandling:
    """Tests for BlockExecutionError handling of None messages."""

    def test_execution_error_with_none_message(self):
        """Test that None message is replaced with descriptive text."""
        error = BlockExecutionError(
            message=None, block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == "Output error was None"

    def test_execution_error_with_valid_message(self):
        """Test that valid messages are preserved."""
        error = BlockExecutionError(
            message="Actual error", block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == "Actual error"

    def test_execution_error_with_empty_string(self):
        """Test that empty string message is NOT replaced (only None is)."""
        error = BlockExecutionError(
            message="", block_name="TestBlock", block_id="test-123"
        )
        assert str(error) == ""


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
