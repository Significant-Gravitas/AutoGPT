"""Linking a Telegram group, end to end in one process.

/setup runs through the real command handler, shared policy and BotBackend,
the real token and link functions, and the real confirm route with real
Telegram login verification. Only the edges are stand-ins: the Bot API, the
linking-manager RPC hop, and Prisma (an in-memory table per model).
"""

import hashlib
import hmac
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlsplit

import pytest
from fastapi import HTTPException

from backend.api.features.platform_linking.routes import (
    ConfirmLinkRequest,
    confirm_link_token,
    confirm_user_link_token,
)
from backend.copilot.bot.bot_backend import BotBackend
from backend.platform_linking import db as linking_db

from . import commands

_BOT_TOKEN = "123456:test-bot-token"
_GROUP_ID = -100123456
_ADMIN_ID = 42
_MEMBER_ID = 77


class _Table:
    """In-memory stand-in for one Prisma model: only the calls db.py makes."""

    def __init__(self) -> None:
        self.rows: list[SimpleNamespace] = []

    def prisma(self, tx: Any = None) -> "_Table":
        return self

    async def find_unique(self, where: dict, include: dict | None = None) -> Any:
        return next((row for row in self.rows if _matches(row, where)), None)

    async def find_first(self, where: dict) -> Any:
        return next((row for row in self.rows if _matches(row, where)), None)

    async def update_many(self, where: dict, data: dict) -> int:
        hits = [row for row in self.rows if _matches(row, where)]
        for row in hits:
            vars(row).update(data)
        return len(hits)

    async def create(self, data: dict) -> SimpleNamespace:
        defaults = {
            "id": f"row-{len(self.rows) + 1}",
            "usedAt": None,
            "linkedAt": datetime.now(UTC),
        }
        row = SimpleNamespace(**{**defaults, **data})
        self.rows.append(row)
        return row


def _matches(row: SimpleNamespace, where: dict) -> bool:
    for key, expected in where.items():
        if key == "platform_platformUserId":  # PlatformUserLink's compound key
            if not _matches(row, expected):
                return False
            continue
        actual = getattr(row, key, None)
        if isinstance(expected, dict):  # {"gt": bound}
            if actual is None or not actual > expected["gt"]:
                return False
        elif actual != expected:
            return False
    return True


@asynccontextmanager
async def _no_transaction():
    yield None


class _LinkingManager:
    """The linking manager's RPC methods BotBackend calls, answered by the
    real db functions, as the manager service answers them."""

    async def create_server_link_token(self, request):
        return await linking_db.create_server_link_token(request)

    async def create_user_link_token(self, request):
        return await linking_db.create_user_link_token(request)

    async def resolve_user_link(self, platform, platform_user_id, include_account):
        return await linking_db.resolve_user_link(
            platform.value, platform_user_id, include_account
        )

    async def record_bot_event(self, event):
        return None


class _TelegramGroup:
    """The Bot API as one group sees it: who administers it, what was posted."""

    def __init__(self, admins: set[int]) -> None:
        self.admins = admins
        self.sent: list[dict] = []
        self.member_lookups = 0
        self.client = MagicMock()
        self.client.call = AsyncMock(side_effect=self._call)

    async def _call(self, method: str, **params: Any) -> Any:
        if method == "getChatMember":
            self.member_lookups += 1
            admin = params["user_id"] in self.admins
            return {"status": "administrator" if admin else "member"}
        if method == "sendMessage":
            self.sent.append(params)
            return {"message_id": len(self.sent)}
        raise AssertionError(f"unexpected Bot API call {method}")


@contextmanager
def _linking_world() -> Iterator[SimpleNamespace]:
    tables = SimpleNamespace(tokens=_Table(), links=_Table(), dm_links=_Table())
    with (
        patch.multiple(
            "backend.platform_linking.db",
            PlatformLinkToken=tables.tokens,
            PlatformLink=tables.links,
            PlatformUserLink=tables.dm_links,
            transaction=_no_transaction,
            _link_base_url=lambda: "https://app.example/link",
        ),
        patch(
            "backend.copilot.bot.bot_backend.get_platform_linking_manager_client",
            return_value=_LinkingManager(),
        ),
        patch(
            "backend.api.features.platform_linking.routes.platform_linking_db",
            return_value=linking_db,
        ),
        patch(
            "backend.copilot.bot.adapters.telegram.config.get_bot_token",
            return_value=_BOT_TOKEN,
        ),
    ):
        yield tables


def _command(sender_id: int, text: str, chat: dict) -> dict:
    return {
        "message_id": 5,
        "text": text,
        "entities": [{"type": "bot_command", "offset": 0, "length": len(text)}],
        "chat": chat,
        "from": {"id": sender_id, "username": f"user{sender_id}"},
    }


def _in_group(sender_id: int) -> dict:
    chat = {"id": _GROUP_ID, "type": "supergroup", "title": "Builders"}
    return _command(sender_id, "/setup", chat)


def _telegram_login(user_id: int) -> dict[str, str]:
    """What Telegram appends to a login_url when ``user_id`` taps it."""
    fields = {
        "id": str(user_id),
        "first_name": f"User {user_id}",
        "auth_date": str(int(time.time())),
    }
    check = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret = hashlib.sha256(_BOT_TOKEN.encode()).digest()
    fields["hash"] = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    return fields


def _token_of(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


@pytest.mark.asyncio
async def test_only_the_admin_who_ran_setup_can_link_the_group():
    with _linking_world() as tables:
        api = BotBackend()
        group = _TelegramGroup(admins={_ADMIN_ID})

        # A member's /setup is refused and mints nothing.
        await commands.handle(
            api, group.client, _in_group(_MEMBER_ID), "setup", AsyncMock()
        )
        assert tables.tokens.rows == []
        assert group.sent[-1]["text"].startswith("Only a group admin can link")

        # The admin's /setup posts a Telegram sign-in button, never a plain link.
        await commands.handle(
            api, group.client, _in_group(_ADMIN_ID), "setup", AsyncMock()
        )
        [[button]] = group.sent[-1]["reply_markup"]["inline_keyboard"]
        assert set(button) == {"text", "login_url"}
        token = _token_of(button["login_url"]["url"])

        # Another member who opens the posted link can't link the group:
        # not with their own Telegram identity, and not with none.
        for body in (
            ConfirmLinkRequest(telegram_auth=_telegram_login(_MEMBER_ID)),
            None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await confirm_link_token(token, "autogpt-member", body=body)
            assert exc_info.value.status_code == 403
        assert tables.links.rows == []

        # The admin's own tap links the group to the admin's account.
        response = await confirm_link_token(
            token,
            "autogpt-admin",
            body=ConfirmLinkRequest(telegram_auth=_telegram_login(_ADMIN_ID)),
        )
        assert response.success
        [link] = tables.links.rows
        assert (link.userId, link.platformServerId, link.ownerPlatformUserId) == (
            "autogpt-admin",
            str(_GROUP_ID),
            str(_ADMIN_ID),
        )


@pytest.mark.asyncio
async def test_a_member_still_links_their_own_dms():
    with _linking_world() as tables:
        api = BotBackend()
        group = _TelegramGroup(admins=set())
        send_link = AsyncMock()
        private = {"id": _MEMBER_ID, "type": "private"}

        await commands.handle(
            api,
            group.client,
            _command(_MEMBER_ID, "/start", private),
            "start",
            send_link,
        )
        target, _text, label, url = send_link.await_args.args
        assert (target, label) == (str(_MEMBER_ID), "Link Account")
        assert group.member_lookups == 0

        # A DM link needs no admin role and no Telegram identity to confirm.
        response = await confirm_user_link_token(
            _token_of(url), "autogpt-member", body=None
        )
        assert response.success
        [dm_link] = tables.dm_links.rows
        assert (dm_link.userId, dm_link.platformUserId) == (
            "autogpt-member",
            str(_MEMBER_ID),
        )
