"""Tests for backend.data.tally module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.tally import (
    _EXTRACTION_PROMPT,
    _EXTRACTION_SUFFIX,
    _build_email_index,
    _format_answer,
    _make_tally_client,
    _mask_email,
    _refresh_cache,
    extract_business_understanding,
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


# ── Prompt construction (curly-brace safety) ─────────────────────────────────


def test_extraction_prompt_safe_with_curly_braces():
    """User content with curly braces must not break prompt construction.

    Previously _EXTRACTION_PROMPT.format(submission_text=...) would raise
    KeyError/ValueError if the user text contained { or }.
    """
    text_with_braces = "Q: What tools do you use?\nA: We use {Slack} and {{Jira}}"
    # This must not raise — the old .format() call would fail here
    prompt = f"{_EXTRACTION_PROMPT}{text_with_braces}{_EXTRACTION_SUFFIX}"
    assert text_with_braces in prompt
    assert prompt.startswith("You are a business analyst.")
    assert prompt.endswith("Return ONLY valid JSON.")


def test_extraction_prompt_no_format_placeholders():
    """_EXTRACTION_PROMPT must not contain Python format placeholders."""
    assert "{submission_text}" not in _EXTRACTION_PROMPT
    # Ensure no stray single-brace placeholders
    # (double braces {{ are fine — they're literal in format strings)
    import re

    single_braces = re.findall(r"(?<!\{)\{[^{].*?\}(?!\})", _EXTRACTION_PROMPT)
    assert single_braces == [], f"Found format placeholders: {single_braces}"


# ── extract_business_understanding ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_business_understanding_success():
    """Happy path: LLM returns valid JSON that maps to BusinessUnderstandingInput."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "user_name": "Alice",
            "business_name": "Acme Corp",
            "industry": "Technology",
            "pain_points": ["manual reporting"],
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("backend.data.tally.AsyncOpenAI", return_value=mock_client):
        result = await extract_business_understanding("Q: Name?\nA: Alice")

    assert result.user_name == "Alice"
    assert result.business_name == "Acme Corp"
    assert result.industry == "Technology"
    assert result.pain_points == ["manual reporting"]


@pytest.mark.asyncio
async def test_extract_business_understanding_filters_nulls():
    """Null values from LLM should be excluded from the result."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {"user_name": "Alice", "business_name": None, "industry": None}
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("backend.data.tally.AsyncOpenAI", return_value=mock_client):
        result = await extract_business_understanding("Q: Name?\nA: Alice")

    assert result.user_name == "Alice"
    assert result.business_name is None
    assert result.industry is None


@pytest.mark.asyncio
async def test_extract_business_understanding_invalid_json():
    """Invalid JSON from LLM should raise JSONDecodeError."""
    mock_choice = MagicMock()
    mock_choice.message.content = "not valid json {"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch("backend.data.tally.AsyncOpenAI", return_value=mock_client),
        pytest.raises(json.JSONDecodeError),
    ):
        await extract_business_understanding("Q: Name?\nA: Alice")


@pytest.mark.asyncio
async def test_extract_business_understanding_timeout():
    """LLM timeout should propagate as asyncio.TimeoutError."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()

    with (
        patch("backend.data.tally.AsyncOpenAI", return_value=mock_client),
        patch("backend.data.tally._LLM_TIMEOUT", 0.001),
        pytest.raises(asyncio.TimeoutError),
    ):
        await extract_business_understanding("Q: Name?\nA: Alice")


# ── _refresh_cache ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_cache_full_fetch():
    """First fetch (no last_fetch in Redis) should do a full fetch and store in Redis."""
    mock_settings = MagicMock()
    mock_settings.secrets.tally_api_key = "test-key"

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # No last_fetch, no cached index

    questions = SAMPLE_QUESTIONS
    submissions = SAMPLE_SUBMISSIONS

    with (
        patch("backend.data.tally.Settings", return_value=mock_settings),
        patch(
            "backend.data.tally.get_redis_async",
            new_callable=AsyncMock,
            return_value=mock_redis,
        ),
        patch(
            "backend.data.tally._fetch_all_submissions",
            new_callable=AsyncMock,
            return_value=(questions, submissions),
        ) as mock_fetch,
    ):
        index, returned_questions = await _refresh_cache("form123")

    mock_fetch.assert_awaited_once()
    assert "alice@example.com" in index
    assert "bob@example.com" in index
    assert returned_questions == questions
    # Verify Redis setex was called for index, questions, and last_fetch
    assert mock_redis.setex.await_count == 3


@pytest.mark.asyncio
async def test_refresh_cache_incremental_fetch():
    """When last_fetch and index both exist, should do incremental fetch and merge."""
    mock_settings = MagicMock()
    mock_settings.secrets.tally_api_key = "test-key"

    existing_index = {
        "old@example.com": {"responses": [], "submitted_at": "2025-01-01"}
    }

    mock_redis = AsyncMock()

    def mock_get(key):
        if "last_fetch" in key:
            return "2025-01-14T00:00:00Z"
        if "email_index" in key:
            return json.dumps(existing_index)
        if "questions" in key:
            return json.dumps(SAMPLE_QUESTIONS)
        return None

    mock_redis.get.side_effect = mock_get

    new_submissions = [SAMPLE_SUBMISSIONS[0]]  # Just Alice

    with (
        patch("backend.data.tally.Settings", return_value=mock_settings),
        patch(
            "backend.data.tally.get_redis_async",
            new_callable=AsyncMock,
            return_value=mock_redis,
        ),
        patch(
            "backend.data.tally._fetch_all_submissions",
            new_callable=AsyncMock,
            return_value=(SAMPLE_QUESTIONS, new_submissions),
        ),
    ):
        index, _ = await _refresh_cache("form123")

    # Should contain both old and new entries
    assert "old@example.com" in index
    assert "alice@example.com" in index


# ── _make_tally_client ───────────────────────────────────────────────────────


def test_make_tally_client_returns_configured_client():
    """_make_tally_client should create a Requests client with auth headers."""
    client = _make_tally_client("test-api-key")
    assert client.extra_headers is not None
    assert client.extra_headers.get("Authorization") == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_fetch_tally_page_uses_provided_client():
    """_fetch_tally_page should use the passed client, not create its own."""
    from backend.data.tally import _fetch_tally_page

    mock_response = MagicMock()
    mock_response.json.return_value = {"submissions": [], "questions": []}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    result = await _fetch_tally_page(mock_client, "form123", page=1)

    mock_client.get.assert_awaited_once()
    call_url = mock_client.get.call_args[0][0]
    assert "form123" in call_url
    assert "page=1" in call_url
    assert result == {"submissions": [], "questions": []}
