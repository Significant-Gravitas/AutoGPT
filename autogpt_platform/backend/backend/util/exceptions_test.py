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
    UserPaywalledError,
    get_execution_failure_reason,
)

_ANALYTICS_QUERIES_DIR = Path(__file__).resolve().parents[3] / "analytics" / "queries"
_GRAPH_EXECUTION_SQL = _ANALYTICS_QUERIES_DIR / "graph_execution.sql"
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
    (
        "Insufficient balance to run Third-party API: "
        "dynamic-cost blocks require a positive balance.",
        False,
    ),
)


def _read_graph_execution_sql() -> str:
    """Load the analytics SQL, failing closed on drift inside a source checkout.

    Skipping is only legitimate when the whole ``analytics/queries`` tree is
    absent, i.e. the backend is running from a packaged install (the runtime
    container ships no analytics SQL). When that directory *is* present the
    parity guarantee is in force, so a missing or renamed file is drift and
    must fail rather than silently dropping the Python<->SQL check.
    """
    if _GRAPH_EXECUTION_SQL.exists():
        return _GRAPH_EXECUTION_SQL.read_text(encoding="utf-8")
    if _ANALYTICS_QUERIES_DIR.is_dir():
        pytest.fail(
            f"{_GRAPH_EXECUTION_SQL} is missing while {_ANALYTICS_QUERIES_DIR} "
            "exists: the Python<->SQL legacy-classifier parity check cannot be "
            "verified. Update the path if the query moved."
        )
    pytest.skip("analytics SQL is not included in this backend package")


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


def test_paywall_error_is_classified_by_type():
    """A paywalled node failure is a trusted, terminal, user-actionable
    failure — the same class as an exhausted wallet. Without a typed reason it
    never reached graph stats, so a denied run reported COMPLETED with a null
    error while the node inside carried the real denial."""
    error = UserPaywalledError("A Max plan or higher is required to use ChatGPT.")

    assert (
        get_execution_failure_reason(error)
        == ExecutionFailureReason.ENTITLEMENT_REQUIRED
    )


def test_paywall_error_is_still_importable_from_rate_limit():
    """It moved to break an import cycle; every existing call site imports it
    from its old home."""
    from backend.copilot.rate_limit import UserPaywalledError as ReExported

    assert ReExported is UserPaywalledError


def test_untyped_entitlement_wording_is_not_classified():
    """Classification is by type, never by message text."""
    assert (
        get_execution_failure_reason(
            RuntimeError("A Max plan or higher is required to use ChatGPT.")
        )
        is None
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
