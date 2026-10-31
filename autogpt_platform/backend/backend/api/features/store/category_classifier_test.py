from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.util.clients import OPENROUTER_BASE_URL

from . import category_classifier
from .categories import StoreCategory
from .category_classifier import _parse_category, classify_category


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("sales", StoreCategory.SALES),
        ("  Sales\n", StoreCategory.SALES),
        ("`content`", StoreCategory.CONTENT),
        ("Development.", StoreCategory.DEVELOPMENT),
        ("none", None),
        ("", None),
        ("writing", None),
        ("sales and marketing", None),
    ],
)
def test_parse_category(answer, expected):
    assert _parse_category(answer) is expected


def _client(content: str):
    client = MagicMock()
    client.base_url = httpx.URL(OPENROUTER_BASE_URL)
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
    )
    return client


@pytest.mark.asyncio
async def test_the_classification_asks_for_no_reasoning(monkeypatch):
    # Haiku 5.5 spent all 16 tokens thinking on 1 listing in 5 and answered
    # nothing, leaving it uncategorised.
    client = _client("sales")
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: client)
    monkeypatch.setattr(category_classifier, "_MODEL", "anthropic/claude-haiku-5.5")

    await category_classifier.classify_category("Lead Finder", "Find prospects", "")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}
    assert kwargs["temperature"] == 0


async def test_classifies_a_listing(monkeypatch):
    client = _client("finance")
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: client)

    assert (
        await classify_category("Invoice Chaser", "Chases invoices", "Emails debtors.")
        is StoreCategory.FINANCE
    )

    prompt = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    for category in StoreCategory:
        assert category.value in prompt
    assert "Invoice Chaser" in prompt


async def test_returns_none_when_the_model_answers_with_no_choices(monkeypatch):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(choices=[]))
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: client)

    assert (
        await classify_category("Invoice Chaser", "Chases invoices", "Emails.") is None
    )


async def test_returns_none_without_a_configured_client(monkeypatch):
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: None)

    assert await classify_category("n", "s", "d") is None


async def test_a_failed_call_never_propagates(monkeypatch):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: client)

    assert await classify_category("n", "s", "d") is None


async def test_a_long_description_is_truncated_before_it_is_sent(monkeypatch):
    client = _client("research")
    monkeypatch.setattr(category_classifier, "get_openai_client", lambda **_: client)

    await classify_category("n", "s", "x" * 5000)

    prompt = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "x" * 2000 in prompt
    assert "x" * 2001 not in prompt
