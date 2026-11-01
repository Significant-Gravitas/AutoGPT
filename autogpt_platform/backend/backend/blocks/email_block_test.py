"""SendEmailBlock only sends a saved SMTP login to the server its scope names;
a login with no scope is only used in a graph version the running user saved."""

import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.email_block import SendEmailBlock, UserPasswordCredentials


def _make_credentials(scope: str | None) -> UserPasswordCredentials:
    return UserPasswordCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="smtp",
        username=SecretStr("user@example.com"),
        password=SecretStr("app-password"),
        scope=scope,
        title="test creds",
    )


def _make_input(smtp_server: str, smtp_port: int = 587) -> SendEmailBlock.Input:
    return SendEmailBlock.Input.model_validate(
        {
            "to_email": "recipient@example.com",
            "subject": "Subject",
            "body": "Body",
            "config": {"smtp_server": smtp_server, "smtp_port": smtp_port},
            "credentials": {
                "provider": "smtp",
                "id": "01234567-89ab-cdef-0123-456789abcdef",
                "type": "user_password",
                "title": "test creds",
            },
        }
    )


async def _run(
    input_data: SendEmailBlock.Input,
    credentials: UserPasswordCredentials,
    *,
    owns_graph: bool = False,
    **run_kwargs: Any,
) -> tuple[dict[str, Any], MagicMock]:
    block = SendEmailBlock()
    send_email = MagicMock(return_value="Email sent successfully")
    db = MagicMock(is_graph_version_owner=AsyncMock(return_value=owns_graph))
    outputs: dict[str, Any] = {}
    with (
        patch.object(SendEmailBlock, "send_email", send_email),
        patch(
            "backend.blocks.email_block.resolve_and_check_blocked",
            new=AsyncMock(return_value=["203.0.113.7"]),
        ),
        patch(
            "backend.blocks.helpers.host_bound_login.get_database_manager_async_client",
            return_value=db,
        ),
    ):
        async for name, value in block.run(
            input_data, credentials=credentials, **run_kwargs
        ):
            outputs[name] = value
    return outputs, send_email


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,smtp_server,smtp_port",
    [
        ("smtp://smtp.example.com", "203.0.113.7", 587),
        (None, "smtp.example.com", 587),
        ("", "smtp.example.com", 587),
        ("smtp.example.com", "smtp.example.com", 587),
        ("smtp://smtp.example.com:465", "smtp.example.com", 587),
        ("smtp://smtp.example.com", "sub.smtp.example.com", 587),
        ("smtp://smtp.example.com", "mail.example.org", 587),
        ("postgresql://smtp.example.com", "smtp.example.com", 587),
    ],
)
async def test_login_not_sent_to_another_server(
    scope: str | None, smtp_server: str, smtp_port: int
):
    outputs, send_email = await _run(
        _make_input(smtp_server, smtp_port), _make_credentials(scope)
    )
    error = outputs["error"]
    if scope is None:
        assert "saved without a server" in error
    else:
        assert f"different server than this workflow's ('{smtp_server}'" in error
    assert "so they were not sent" in error
    assert not re.search(r"\badd\b", error, re.IGNORECASE)
    assert "status" not in outputs
    send_email.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,smtp_server,smtp_port",
    [
        ("smtp://smtp.example.com", "smtp.example.com", 587),
        ("smtp://SMTP.Example.com:587", "smtp.example.com", 587),
        ("smtp://203.0.113.7", "203.0.113.7", 2525),
    ],
)
async def test_login_sent_to_its_own_server(
    scope: str, smtp_server: str, smtp_port: int
):
    outputs, send_email = await _run(
        _make_input(smtp_server, smtp_port), _make_credentials(scope)
    )
    assert outputs == {"status": "Email sent successfully"}
    send_email.assert_called_once()


_RUN_BY_USER = {"user_id": "user-1", "graph_id": "graph-1", "graph_version": 3}


@pytest.mark.asyncio
async def test_login_without_scope_used_on_own_graph():
    """A login with no scope is used in a graph version the running user saved."""
    outputs, send_email = await _run(
        _make_input("smtp.example.com"),
        _make_credentials(None),
        owns_graph=True,
        **_RUN_BY_USER,
    )
    assert outputs == {"status": "Email sent successfully"}
    send_email.assert_called_once()


@pytest.mark.asyncio
async def test_login_without_scope_not_used_on_another_users_graph():
    """A login with no scope is not used in a graph version another user saved."""
    outputs, send_email = await _run(
        _make_input("203.0.113.7"),
        _make_credentials(None),
        owns_graph=False,
        **_RUN_BY_USER,
    )
    assert "saved without a server" in outputs["error"]
    send_email.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["smtp://smtp.example.com", ""])
async def test_login_with_scope_ignores_graph_ownership(scope: str):
    """A login with a scope goes only where its scope says, even on the user's
    own graph; only a login with no scope at all is a legacy one."""
    outputs, send_email = await _run(
        _make_input("203.0.113.7"),
        _make_credentials(scope),
        owns_graph=True,
        **_RUN_BY_USER,
    )
    assert "different server" in outputs["error"]
    send_email.assert_not_called()
