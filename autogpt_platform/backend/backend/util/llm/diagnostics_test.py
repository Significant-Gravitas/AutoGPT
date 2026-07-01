from unittest.mock import AsyncMock, patch

import pytest

from backend.util.llm.diagnostics import diagnose_chat_provider
from backend.util.llm.providers import ProviderResponse


@pytest.mark.asyncio
async def test_diagnostic_uses_resolved_deepseek_config() -> None:
    with patch(
        "backend.util.llm.diagnostics.call_provider",
        new=AsyncMock(
            return_value=ProviderResponse(
                content="ok",
                prompt_tokens=1,
                completion_tokens=1,
            )
        ),
    ) as call:
        result = await diagnose_chat_provider(
            {
                "CHAT_PROVIDER": "deepseek",
                "CHAT_API_KEY": "secret",
            }
        )

    assert result.success is True
    assert result.provider == "deepseek"
    assert result.model == "deepseek-v4-flash"
    assert result.base_url == "https://api.deepseek.com"
    assert call.await_args.kwargs["api_key"] == "secret"
    assert call.await_args.kwargs["max_retries"] == 1


@pytest.mark.asyncio
async def test_diagnostic_never_returns_api_key_in_error() -> None:
    with patch(
        "backend.util.llm.diagnostics.call_provider",
        new=AsyncMock(side_effect=RuntimeError("failed with secret")),
    ):
        result = await diagnose_chat_provider(
            {
                "CHAT_PROVIDER": "deepseek",
                "CHAT_API_KEY": "secret",
            }
        )

    assert result.success is False
    assert "secret" not in (result.error or "")
