"""Tests for backend.data.tally module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.tally import (
    _build_email_index,
    _format_answer,
    _mask_email,
    find_submission_by_email,
    format_submission_for_llm,
    populate_understanding_from_tally,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    {"id": "q1", "label": "What is your name?", "type": "INPUT_TEXT"},
    {"id": "q2", "label": "Email address", "type": "INPUT_EMAIL"},
    {"id": "q3", "label": "Company name", "type": "INPUT_TEXT"},
    {"id": "q4", "label": "Industry", "type": "INPUT_TEXT"},
]

SAMPLE_SUBMISSIONS = [
    {
        "respondentEmail": None,
        "responses": [
            {"questionId": "q1", "value": "Alice Smith"},
            {"questionId": "q2", "value": "alice@example.com"},
            {"questionId": "q3", "value": "Acme Corp"},
            {"questionId": "q4", "value": "Technology"},
        ],
        "submittedAt": "2025-01-15T10:00:00Z",
    },
    {
        "respondentEmail": "bob@example.com",
        "responses": [
            {"questionId": "q1", "value": "Bob Jones"},
            {"questionId": "q2", "value": "bob@example.com"},
            {"questionId": "q3", "value": "Bob's Burgers"},
            {"questionId": "q4", "value": "Food"},
        ],
        "submittedAt": "2025-01-16T10:00:00Z",
    },
]


# ── _build_email_index ────────────────────────────────────────────────────────


def test_build_email_index():
    index = _build_email_index(SAMPLE_SUBMISSIONS, SAMPLE_QUESTIONS)
    assert "alice@example.com" in index
    assert "bob@example.com" in index
    assert len(index) == 2


def test_build_email_index_case_insensitive():
    submissions = [
        {
            "respondentEmail": None,
            "responses": [
                {"questionId": "q2", "value": "Alice@Example.COM"},
            ],
            "submittedAt": "2025-01-15T10:00:00Z",
        },
    ]
    index = _build_email_index(submissions, SAMPLE_QUESTIONS)
    assert "alice@example.com" in index
    assert "Alice@Example.COM" not in index


def test_build_email_index_empty():
    index = _build_email_index([], SAMPLE_QUESTIONS)
    assert index == {}


def test_build_email_index_no_email_field():
    questions = [{"id": "q1", "label": "Name", "type": "INPUT_TEXT"}]
    submissions = [
        {
            "responses": [{"questionId": "q1", "value": "Alice"}],
            "submittedAt": "2025-01-15T10:00:00Z",
        }
    ]
    index = _build_email_index(submissions, questions)
    assert index == {}


def test_build_email_index_respondent_email():
    """respondentEmail takes precedence over field scanning."""
    submissions = [
        {
            "respondentEmail": "direct@example.com",
            "responses": [
                {"questionId": "q2", "value": "field@example.com"},
            ],
            "submittedAt": "2025-01-15T10:00:00Z",
        }
    ]
    index = _build_email_index(submissions, SAMPLE_QUESTIONS)
    assert "direct@example.com" in index
    assert "field@example.com" not in index


# ── format_submission_for_llm ─────────────────────────────────────────────────


def test_format_submission_for_llm():
    submission = {
        "responses": [
            {"questionId": "q1", "value": "Alice Smith"},
            {"questionId": "q3", "value": "Acme Corp"},
        ],
    }
    result = format_submission_for_llm(submission, SAMPLE_QUESTIONS)
    assert "Q: What is your name?" in result
    assert "A: Alice Smith" in result
    assert "Q: Company name" in result
    assert "A: Acme Corp" in result


def test_format_submission_for_llm_dict_responses():
    submission = {
        "responses": {
            "q1": "Alice Smith",
            "q3": "Acme Corp",
        },
    }
    result = format_submission_for_llm(submission, SAMPLE_QUESTIONS)
    assert "A: Alice Smith" in result
    assert "A: Acme Corp" in result


def test_format_answer_types():
    assert _format_answer(None) == "(no answer)"
    assert _format_answer("hello") == "hello"
    assert _format_answer(["a", "b"]) == "a, b"
    assert _format_answer({"key": "val"}) == "key: val"
    assert _format_answer(42) == "42"


# ── find_submission_by_email ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_submission_by_email_cache_hit():
    cached_index = {
        "alice@example.com": {"responses": [], "submitted_at": "2025-01-15"},
    }
    cached_questions = SAMPLE_QUESTIONS

    with patch(
        "backend.data.tally._get_cached_index",
        new_callable=AsyncMock,
        return_value=(cached_index, cached_questions),
    ) as mock_cache:
        result = await find_submission_by_email("form123", "alice@example.com")

    mock_cache.assert_awaited_once_with("form123")
    assert result is not None
    sub, questions = result
    assert sub["submitted_at"] == "2025-01-15"


@pytest.mark.asyncio
async def test_find_submission_by_email_cache_miss():
    refreshed_index = {
        "alice@example.com": {"responses": [], "submitted_at": "2025-01-15"},
    }

    with (
        patch(
            "backend.data.tally._get_cached_index",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch(
            "backend.data.tally._refresh_cache",
            new_callable=AsyncMock,
            return_value=(refreshed_index, SAMPLE_QUESTIONS),
        ) as mock_refresh,
    ):
        result = await find_submission_by_email("form123", "alice@example.com")

    mock_refresh.assert_awaited_once_with("form123")
    assert result is not None


@pytest.mark.asyncio
async def test_find_submission_by_email_no_match():
    cached_index = {
        "alice@example.com": {"responses": [], "submitted_at": "2025-01-15"},
    }

    with patch(
        "backend.data.tally._get_cached_index",
        new_callable=AsyncMock,
        return_value=(cached_index, SAMPLE_QUESTIONS),
    ):
        result = await find_submission_by_email("form123", "unknown@example.com")

    assert result is None


# ── populate_understanding_from_tally ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_populate_understanding_skips_existing():
    """If user already has understanding, skip entirely."""
    mock_understanding = MagicMock()

    with (
        patch(
            "backend.data.tally.get_business_understanding",
            new_callable=AsyncMock,
            return_value=mock_understanding,
        ) as mock_get,
        patch(
            "backend.data.tally.find_submission_by_email",
            new_callable=AsyncMock,
        ) as mock_find,
    ):
        await populate_understanding_from_tally("user-1", "test@example.com")

    mock_get.assert_awaited_once_with("user-1")
    mock_find.assert_not_awaited()


@pytest.mark.asyncio
async def test_populate_understanding_skips_no_api_key():
    """If no Tally API key, skip gracefully."""
    mock_settings = MagicMock()
    mock_settings.secrets.tally_api_key = ""

    with (
        patch(
            "backend.data.tally.get_business_understanding",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("backend.data.tally.Settings", return_value=mock_settings),
        patch(
            "backend.data.tally.find_submission_by_email",
            new_callable=AsyncMock,
        ) as mock_find,
    ):
        await populate_understanding_from_tally("user-1", "test@example.com")

    mock_find.assert_not_awaited()


@pytest.mark.asyncio
async def test_populate_understanding_handles_errors():
    """Must never raise, even on unexpected errors."""
    with patch(
        "backend.data.tally.get_business_understanding",
        new_callable=AsyncMock,
        side_effect=RuntimeError("DB down"),
    ):
        # Should not raise
        await populate_understanding_from_tally("user-1", "test@example.com")


@pytest.mark.asyncio
async def test_populate_understanding_full_flow():
    """Happy path: no existing understanding, finds submission, extracts, upserts."""
    mock_settings = MagicMock()
    mock_settings.secrets.tally_api_key = "test-key"

    submission = {
        "responses": [
            {"questionId": "q1", "value": "Alice"},
            {"questionId": "q3", "value": "Acme"},
        ],
    }
    mock_input = MagicMock()

    with (
        patch(
            "backend.data.tally.get_business_understanding",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("backend.data.tally.Settings", return_value=mock_settings),
        patch(
            "backend.data.tally.find_submission_by_email",
            new_callable=AsyncMock,
            return_value=(submission, SAMPLE_QUESTIONS),
        ),
        patch(
            "backend.data.tally.extract_business_understanding",
            new_callable=AsyncMock,
            return_value=mock_input,
        ) as mock_extract,
        patch(
            "backend.data.tally.upsert_business_understanding",
            new_callable=AsyncMock,
        ) as mock_upsert,
    ):
        await populate_understanding_from_tally("user-1", "alice@example.com")

    mock_extract.assert_awaited_once()
    mock_upsert.assert_awaited_once_with("user-1", mock_input)


@pytest.mark.asyncio
async def test_populate_understanding_handles_llm_timeout():
    """LLM timeout is caught and doesn't raise."""
    import asyncio

    mock_settings = MagicMock()
    mock_settings.secrets.tally_api_key = "test-key"

    submission = {
        "responses": [{"questionId": "q1", "value": "Alice"}],
    }

    with (
        patch(
            "backend.data.tally.get_business_understanding",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("backend.data.tally.Settings", return_value=mock_settings),
        patch(
            "backend.data.tally.find_submission_by_email",
            new_callable=AsyncMock,
            return_value=(submission, SAMPLE_QUESTIONS),
        ),
        patch(
            "backend.data.tally.extract_business_understanding",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ),
        patch(
            "backend.data.tally.upsert_business_understanding",
            new_callable=AsyncMock,
        ) as mock_upsert,
    ):
        await populate_understanding_from_tally("user-1", "alice@example.com")

    mock_upsert.assert_not_awaited()


# ── _mask_email ───────────────────────────────────────────────────────────────


def test_mask_email():
    assert _mask_email("alice@example.com") == "a***e@example.com"
    assert _mask_email("ab@example.com") == "a***@example.com"
    assert _mask_email("a@example.com") == "a***@example.com"


def test_mask_email_invalid():
    assert _mask_email("no-at-sign") == "***"
