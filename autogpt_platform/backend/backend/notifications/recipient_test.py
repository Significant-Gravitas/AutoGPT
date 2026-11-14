from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.notifications import recipient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,email,expected",
    [
        ("Sam Carter", "sam@example.com", "Sam"),
        # Sign-up stores no name unless one is given, so this one was typed.
        ("john", "john@example.com", "john"),
        ("", "jane@example.com", "there"),
        (None, "jane@example.com", "there"),
    ],
)
async def test_greeting_uses_only_the_first_name(monkeypatch, name, email, expected):
    client = SimpleNamespace(
        get_user_by_id=AsyncMock(return_value=SimpleNamespace(name=name, email=email))
    )
    monkeypatch.setattr(
        recipient, "get_database_manager_async_client", lambda **kw: client
    )
    assert await recipient.greeting_name("user-1") == expected
    client.get_user_by_id.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_missing_name_does_not_block_delivery(monkeypatch):
    client = SimpleNamespace(get_user_by_id=AsyncMock(side_effect=TimeoutError))
    monkeypatch.setattr(
        recipient, "get_database_manager_async_client", lambda **kw: client
    )
    assert await recipient.greeting_name("user-1") == "there"
