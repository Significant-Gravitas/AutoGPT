"""Tests for backend.data.understanding_prompts."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.understanding import BusinessUnderstanding
from backend.data.understanding_prompts import generate_understanding_prompts


def make_understanding(**overrides) -> BusinessUnderstanding:
    data = {
        "id": "understanding-1",
        "user_id": "user-1",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "industry": "Customer support",
        "pain_points": ["manual ticket triage"],
        "automation_goals": ["speed up support responses"],
    }
    data.update(overrides)
    return BusinessUnderstanding(**data)


@pytest.mark.asyncio
async def test_generate_understanding_prompts_success():
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "prompts": [
                "Help me automate customer support triage",
                "Show me how to speed up support replies",
                "Find repetitive work in our support process",
            ]
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch(
        "backend.data.understanding_prompts.AsyncOpenAI", return_value=mock_client
    ):
        prompts = await generate_understanding_prompts(make_understanding())

    assert prompts == [
        "Help me automate customer support triage",
        "Show me how to speed up support replies",
        "Find repetitive work in our support process",
    ]


@pytest.mark.asyncio
async def test_generate_understanding_prompts_rejects_duplicates():
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "prompts": [
                "Help me automate customer support",
                "Help me automate customer support",
                "Find repetitive support work",
            ]
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch(
            "backend.data.understanding_prompts.AsyncOpenAI", return_value=mock_client
        ),
        pytest.raises(ValueError, match="unique"),
    ):
        await generate_understanding_prompts(make_understanding())


@pytest.mark.asyncio
async def test_generate_understanding_prompts_rejects_long_prompt():
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "prompts": [
                "Please help me automate every part of our customer support workflow starting with ticket triage routing follow-up escalation and reporting today",
                "Show me better support workflows",
                "Find support busywork for me",
            ]
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch(
            "backend.data.understanding_prompts.AsyncOpenAI", return_value=mock_client
        ),
        pytest.raises(ValueError, match="fewer than 20 words"),
    ):
        await generate_understanding_prompts(make_understanding())


@pytest.mark.asyncio
async def test_generate_understanding_prompts_rejects_invalid_shape():
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {"prompts": ["Help me automate support", "Find repetitive work"]}
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch(
            "backend.data.understanding_prompts.AsyncOpenAI", return_value=mock_client
        ),
        pytest.raises(ValueError, match="exactly three prompts"),
    ):
        await generate_understanding_prompts(make_understanding())


@pytest.mark.asyncio
async def test_generate_understanding_prompts_timeout():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = asyncio.TimeoutError()

    with (
        patch(
            "backend.data.understanding_prompts.AsyncOpenAI", return_value=mock_client
        ),
        patch("backend.data.understanding_prompts._LLM_TIMEOUT", 0.001),
        pytest.raises(asyncio.TimeoutError),
    ):
        await generate_understanding_prompts(make_understanding())
