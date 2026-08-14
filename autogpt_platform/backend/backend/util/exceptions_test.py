import re
from pathlib import Path

import pytest

from backend.util.exceptions import (
    BlockError,
    BlockExecutionError,
    BlockInputError,
    BlockOutputError,
    BlockUnknownError,
    ExecutionFailureReason,
    InsufficientBalanceError,
    get_execution_failure_reason,
)

_GRAPH_EXECUTION_SQL = (
    Path(__file__).resolve().parents[3]
    / "analytics"
    / "queries"
    / "graph_execution.sql"
)
_SQL_LEGACY_ERROR_PREDICATE = re.compile(
    r"""ge\."stats"::jsonb->>'error'\s*(?P<operator>=|~)\s*'(?P<pattern>(?:''|[^'])*)'"""
)
_SQL_NO_CREDITS_BRANCH = re.compile(
    r"""WHEN\s+CAST\(ge\."executionStatus"\s+AS\s+TEXT\)\s*=\s*'FAILED'\s+AND\s*\(\s*ge\."stats"::jsonb->>'failure_reason'\s*=\s*'insufficient_balance'.*?\)\s*THEN\s+'NO_CREDITS'""",
    re.DOTALL,
)
_LEGACY_BALANCE_CORPUS = (
    ("You have no credits left to run an agent.", True),
    ("Insufficient balance of $0.0, where this will cost $1.25", True),
    ("Insufficient balance of $-1.5, where this will cost $0", True),
    (
        "Insufficient balance to run ReplicateModelBlock: "
        "dynamic-cost blocks require a positive balance.",
        True,
    ),
    ("Organization has 12 credits but needs 25", True),
    ("Organization has -5 credits but needs 25", True),
    ("Third-party API reported insufficient balance", False),
    ("Insufficient balance", False),
    ("prefix: You have no credits left to run an agent.", False),
    ("You have no credits left to run an agent. Please retry.", False),
    ("Organization has some credits but needs more", False),
    ("Organization has 12 credits but needs 25.", False),
    (
        "Insufficient balance to run : "
        "dynamic-cost blocks require a positive balance.",
        False,
    ),
)


def _read_graph_execution_sql() -> str:
    if not _GRAPH_EXECUTION_SQL.exists():
        pytest.skip("analytics SQL is not included in this backend package")
    return _GRAPH_EXECUTION_SQL.read_text(encoding="utf-8")


def _matches_analytics_sql_legacy_predicate(message: str) -> bool:
    sql = _read_graph_execution_sql()
    predicates = [
        (match["operator"], match["pattern"].replace("''", "'"))
        for match in _SQL_LEGACY_ERROR_PREDICATE.finditer(sql)
    ]
    assert len(predicates) == 4, "Expected all four legacy SQL error predicates"

    for operator, pattern in predicates:
        if operator == "=" and message == pattern:
            return True
        if operator == "~":
            assert pattern.startswith("^") and pattern.endswith("$")
            if re.fullmatch(pattern, message):
                return True
    return False


def test_typed_insufficient_balance_error_is_classified_by_type():
    error = InsufficientBalanceError(
        message="New producer wording without legacy keywords",
        user_id="user-1",
        balance=0,
        amount=1,
    )

    assert (
        get_execution_failure_reason(error)
        == ExecutionFailureReason.INSUFFICIENT_BALANCE
    )


def test_analytics_sql_gates_structured_credit_failures_on_failed_status():
    sql = _read_graph_execution_sql()

    assert _SQL_NO_CREDITS_BRANCH.search(sql)


def test_untyped_balance_error_is_not_classified():
    assert (
        get_execution_failure_reason(
            RuntimeError("New producer wording without legacy keywords")
        )
        is None
    )


@pytest.mark.parametrize(("message", "expected"), _LEGACY_BALANCE_CORPUS)
def test_legacy_credit_classifier_matches_analytics_sql(message, expected):
    assert get_execution_failure_reason(message) is None
    python_matches = (
        get_execution_failure_reason(message, allow_legacy_text=True)
        == ExecutionFailureReason.INSUFFICIENT_BALANCE
    )
    sql_matches = _matches_analytics_sql_legacy_predicate(message)

    assert python_matches is expected
    assert sql_matches is expected


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
