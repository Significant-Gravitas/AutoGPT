"""Route tests: domain exceptions → HTTPException status codes."""

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import pytest
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.features.platform_linking.registry import PlatformMeta
from backend.api.features.platform_linking.routes import (
    ConfirmLinkRequest,
    confirm_link_token,
    confirm_user_link_token,
    delete_link,
    delete_user_link_route,
    get_link_token_info_route,
    list_bot_platforms,
    router,
)
from backend.copilot.bot.adapters.slack.pending import PendingSlackInstall
from backend.platform_linking.models import (
    ConfirmLinkResponse,
    ConfirmUserLinkResponse,
    PlatformLinkInfo,
    PlatformUserLinkInfo,
)
from backend.util.exceptions import (
    LinkAlreadyExistsError,
    LinkFlowMismatchError,
    LinkTokenExpiredError,
    NotAuthorizedError,
    NotFoundError,
)

_R = "backend.api.features.platform_linking.routes"


class TestTokenInfoRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (LinkTokenExpiredError("expired"), 410),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        with _patched_db(get_link_token_info=AsyncMock(side_effect=exc)):
            with pytest.raises(HTTPException) as ctx:
                await get_link_token_info_route(token="abc")
        assert ctx.value.status_code == expected_status


class TestConfirmLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (LinkFlowMismatchError("wrong flow"), 400),
            (LinkTokenExpiredError("expired"), 410),
            (LinkAlreadyExistsError("already"), 409),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        with _patched_db(confirm_server_link=AsyncMock(side_effect=exc)):
            with pytest.raises(HTTPException) as ctx:
                await confirm_link_token(token="abc", user_id="u1")
        assert ctx.value.status_code == expected_status


class TestConfirmUserLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (LinkFlowMismatchError("wrong flow"), 400),
            (LinkTokenExpiredError("expired"), 410),
            (LinkAlreadyExistsError("already"), 409),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        with _patched_db(confirm_user_link=AsyncMock(side_effect=exc)):
            with pytest.raises(HTTPException) as ctx:
                await confirm_user_link_token(token="abc", user_id="u1")
        assert ctx.value.status_code == expected_status


class TestDeleteLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (NotAuthorizedError("nope"), 403),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        with _patched_db(delete_server_link=AsyncMock(side_effect=exc)):
            with pytest.raises(HTTPException) as ctx:
                await delete_link(link_id="x", user_id="u1")
        assert ctx.value.status_code == expected_status


class TestDeleteUserLinkRouteTranslation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc,expected_status",
        [
            (NotFoundError("missing"), 404),
            (NotAuthorizedError("nope"), 403),
        ],
    )
    async def test_translation(self, exc: Exception, expected_status: int):
        with _patched_db(delete_user_link=AsyncMock(side_effect=exc)):
            with pytest.raises(HTTPException) as ctx:
                await delete_user_link_route(link_id="x", user_id="u1")
        assert ctx.value.status_code == expected_status


# ── Adversarial: malformed token path params ──────────────────────────


class TestAdversarialTokenPath:
    # TokenPath enforces `^[A-Za-z0-9_-]+$` + max_length=64.

    def test_rejects_token_with_special_chars(self, client):
        response = client.get("/api/platform-linking/tokens/bad%24token/info")
        assert response.status_code == 422

    def test_rejects_token_with_path_traversal(self, client):
        for probe in ("..%2F..", "foo..bar", "foo%2Fbar"):
            response = client.get(f"/api/platform-linking/tokens/{probe}/info")
            assert response.status_code in (
                404,
                422,
            ), f"path-traversal probe {probe!r} returned {response.status_code}"

    def test_rejects_token_too_long(self, client):
        long_token = "a" * 65
        response = client.get(f"/api/platform-linking/tokens/{long_token}/info")
        assert response.status_code == 422

    def test_accepts_token_at_max_length(self, client):
        token = "a" * 64
        with _patched_db(
            get_link_token_info=AsyncMock(side_effect=NotFoundError("missing"))
        ):
            response = client.get(f"/api/platform-linking/tokens/{token}/info")
        assert response.status_code == 404

    def test_accepts_urlsafe_b64_token_shape(self, client):
        with _patched_db(
            get_link_token_info=AsyncMock(side_effect=NotFoundError("missing"))
        ):
            response = client.get("/api/platform-linking/tokens/abc-_XYZ123-_abc/info")
        assert response.status_code == 404

    def test_confirm_rejects_malformed_token(self, client):
        response = client.post("/api/platform-linking/tokens/bad%24token/confirm")
        assert response.status_code == 422


class TestAdversarialDeleteLinkId:
    """DELETE link_id has no regex — ensure weird values are handled via
    NotFoundError (no crash, no cross-user leak)."""

    def test_weird_link_id_returns_404(self, client):
        with _patched_db(
            delete_server_link=AsyncMock(side_effect=NotFoundError("missing"))
        ):
            for link_id in ("'; DROP TABLE links;--", "../../etc/passwd", ""):
                response = client.delete(f"/api/platform-linking/links/{link_id}")
                assert response.status_code in (404, 405)


class TestListBotPlatforms:
    @pytest.mark.asyncio
    async def test_passes_the_dm_url_through(self):
        telegram = PlatformMeta(
            platform="TELEGRAM",
            display_name="Telegram",
            icon="telegram.png",
            server_noun="group",
            enabled=True,
            add_bot_url="https://t.me/AutoGPTBot?startgroup=true",
            dm_url="https://t.me/AutoGPTBot?start=connect",
        )
        with _patched_listing([telegram, _discord_meta()]):
            result = await list_bot_platforms(user_id="u1")

        assert result[0].dm_url == "https://t.me/AutoGPTBot?start=connect"
        assert result[0].add_bot_url == "https://t.me/AutoGPTBot?startgroup=true"
        assert result[1].dm_url is None

    @pytest.mark.asyncio
    async def test_returns_enabled_platforms_with_no_links(self):
        with _patched_listing([_discord_meta()]):
            result = await list_bot_platforms(user_id="u1")

        assert len(result) == 1
        assert result[0].platform == "DISCORD"
        assert result[0].add_bot_url == "https://invite"
        assert result[0].dm_link is None
        assert result[0].server_links == []

    @pytest.mark.asyncio
    async def test_attaches_callers_dm_and_server_links(self):
        dm_link = PlatformUserLinkInfo(
            id="dm-1",
            platform="DISCORD",
            platform_user_id="discord-user-1",
            platform_username="bently",
            linked_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        server_a = PlatformLinkInfo(
            id="srv-a",
            platform="DISCORD",
            platform_server_id="111",
            owner_platform_user_id="discord-user-1",
            server_name="Server A",
            linked_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        server_b = PlatformLinkInfo(
            id="srv-b",
            platform="DISCORD",
            platform_server_id="222",
            owner_platform_user_id="discord-user-1",
            server_name="Server B",
            linked_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        with _patched_listing(
            [_discord_meta()], user_links=[dm_link], server_links=[server_a, server_b]
        ):
            result = await list_bot_platforms(user_id="discord-user-1")

        assert result[0].dm_link is not None
        assert result[0].dm_link.id == "dm-1"
        assert [s.id for s in result[0].server_links] == ["srv-a", "srv-b"]

    @pytest.mark.asyncio
    async def test_picks_newest_dm_link_when_multiple_for_same_platform(self):
        newer = PlatformUserLinkInfo(
            id="dm-new",
            platform="DISCORD",
            platform_user_id="discord-user-2",
            platform_username="newer-handle",
            linked_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        )
        older = PlatformUserLinkInfo(
            id="dm-old",
            platform="DISCORD",
            platform_user_id="discord-user-1",
            platform_username="older-handle",
            linked_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        with _patched_listing([_discord_meta()], user_links=[newer, older]):
            result = await list_bot_platforms(user_id="u1")

        assert result[0].dm_link is not None
        assert result[0].dm_link.id == "dm-new"

    @pytest.mark.asyncio
    async def test_omits_platforms_whose_adapter_isnt_configured(self):
        with _patched_listing([]):
            result = await list_bot_platforms(user_id="u1")

        assert result == []


class TestTelegramLoginVerification:
    @pytest.mark.asyncio
    async def test_invalid_telegram_auth_is_403_before_any_db_call(self):
        confirm = AsyncMock()
        with (
            _patched_db(confirm_user_link=confirm),
            patch(f"{_R}.verify_login", return_value=None),
        ):
            with pytest.raises(HTTPException) as exc:
                await confirm_user_link_token(
                    "tok",
                    "user-1",
                    body=ConfirmLinkRequest(telegram_auth={"id": "1", "hash": "x"}),
                )
        assert exc.value.status_code == 403
        confirm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_verified_identity_is_forwarded_to_confirm(self):
        confirm = AsyncMock(return_value=MagicMock())
        with (
            _patched_db(confirm_user_link=confirm),
            patch(f"{_R}.verify_login", return_value="424242"),
        ):
            await confirm_user_link_token(
                "tok",
                "user-1",
                body=ConfirmLinkRequest(telegram_auth={"id": "424242", "hash": "x"}),
            )
        confirm.assert_awaited_once_with(
            "tok", "user-1", verified_platform_user_id="424242"
        )

    @pytest.mark.asyncio
    async def test_no_body_confirms_without_verification(self):
        confirm = AsyncMock(return_value=MagicMock())
        with _patched_db(confirm_user_link=confirm):
            await confirm_user_link_token("tok", "user-1", body=None)
        confirm.assert_awaited_once_with(
            "tok", "user-1", verified_platform_user_id=None
        )

    @pytest.mark.asyncio
    async def test_identity_mismatch_from_db_maps_to_403(self):
        confirm = AsyncMock(side_effect=NotAuthorizedError("different user"))
        with _patched_db(confirm_user_link=confirm):
            with pytest.raises(HTTPException) as exc:
                await confirm_user_link_token("tok", "user-1", body=None)
        assert exc.value.status_code == 403


class TestClosedLoopSlackFlow:
    @pytest.mark.asyncio
    async def test_slack_install_url_is_user_bound(self):
        with (
            _patched_listing([_slack_meta()]),
            patch(
                f"{_R}.slack_oauth.make_install_user_param",
                return_value="u1.1.sig",
            ),
            patch(f"{_R}.slack_pending.get_pending", new=AsyncMock(return_value=None)),
        ):
            result = await list_bot_platforms(user_id="u1")

        assert result[0].add_bot_url == (
            "https://b.example/api/copilot-webhooks/slack/install?u=u1.1.sig"
        )
        assert result[0].pending_install is None

    @pytest.mark.asyncio
    async def test_pending_install_surfaces_until_dm_linked(self):
        marker = PendingSlackInstall(team_id="T1", team_name="eek", app_id="A1")
        with (
            _patched_listing([_slack_meta()]),
            patch(
                f"{_R}.slack_pending.get_pending", new=AsyncMock(return_value=marker)
            ),
        ):
            result = await list_bot_platforms(user_id="u1")

        pending = result[0].pending_install
        assert pending is not None
        assert pending.server_name == "eek"
        assert pending.open_bot_url == "https://slack.com/app_redirect?app=A1&team=T1"

    @pytest.mark.asyncio
    async def test_user_confirm_returns_deep_link_and_clears_marker(self):
        marker = PendingSlackInstall(team_id="T1", team_name="eek", app_id="A1")
        with (
            _patched_user_confirm(platform="SLACK", platform_user_id="U1"),
            patch(
                f"{_R}.slack_pending.get_pending", new=AsyncMock(return_value=marker)
            ),
            patch(f"{_R}.slack_pending.clear_pending", new=AsyncMock()) as cleared,
        ):
            response = await confirm_user_link_token("tok", "u1", body=None)

        assert response.return_url == "https://slack.com/app_redirect?app=A1&team=T1"
        cleared.assert_awaited_once_with("u1")

    @pytest.mark.asyncio
    async def test_server_confirm_returns_deep_link_from_install(self):
        install = MagicMock()
        install.app_id = "A1"
        with _patched_slack_server_confirm(install):
            response = await confirm_link_token("tok", "u1", body=None)

        assert response.return_url == "https://slack.com/app_redirect?app=A1&team=T1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("install", [None, "no-app-id"])
    async def test_server_confirm_has_no_deep_link_without_a_usable_install(
        self, install
    ):
        if install == "no-app-id":
            install = MagicMock()
            install.app_id = None
        with _patched_slack_server_confirm(install):
            response = await confirm_link_token("tok", "u1", body=None)

        # No app id means no deep link we can build; the link itself still stands.
        assert response.success is True
        assert response.return_url is None

    @pytest.mark.asyncio
    async def test_user_confirm_without_a_marker_still_clears_and_returns_no_link(self):
        with (
            _patched_user_confirm(platform="SLACK", platform_user_id="U1"),
            patch(f"{_R}.slack_pending.get_pending", new=AsyncMock(return_value=None)),
            patch(f"{_R}.slack_pending.clear_pending", new=AsyncMock()) as cleared,
        ):
            response = await confirm_user_link_token("tok", "u1", body=None)

        assert response.return_url is None
        # Clearing is unconditional: a marker may exist even when this call
        # could not read one back.
        cleared.assert_awaited_once_with("u1")

    @pytest.mark.asyncio
    async def test_non_slack_platforms_are_untouched(self):
        with _patched_listing([_discord_meta()]):
            result = await list_bot_platforms(user_id="u1")

        assert result[0].add_bot_url == "https://invite"
        assert result[0].pending_install is None

    @pytest.mark.asyncio
    async def test_telegram_user_confirm_returns_tme_link(self):
        with (
            _patched_user_confirm(platform="TELEGRAM", platform_user_id="tg-1"),
            patch(
                f"{_R}.telegram_config.get_bot_username",
                return_value="AutoGPTBot",
            ),
        ):
            response = await confirm_user_link_token("tok", "u1", body=None)

        assert response.return_url == "https://t.me/AutoGPTBot"


@pytest.fixture
def client():
    app = fastapi.FastAPI()
    app.dependency_overrides[requires_user] = lambda: None
    app.dependency_overrides[get_user_id] = lambda: "caller-user"
    app.include_router(router, prefix="/api/platform-linking")
    return TestClient(app)


def _patched_db(**methods):
    """Patch the routes' database accessor to return a mock with these methods."""
    db = MagicMock()
    for name, method in methods.items():
        setattr(db, name, method)
    return patch(f"{_R}.platform_linking_db", return_value=db)


@contextmanager
def _patched_listing(platforms, *, user_links=None, server_links=None):
    """Patch what ``list_bot_platforms`` reads: the enabled platforms and the
    caller's DM and server links (none unless given)."""
    with (
        _patched_db(
            list_user_links=AsyncMock(return_value=user_links or []),
            list_server_links=AsyncMock(return_value=server_links or []),
        ),
        patch(f"{_R}.registry.enabled_platforms", return_value=platforms),
    ):
        yield


def _patched_user_confirm(*, platform: str, platform_user_id: str):
    return _patched_db(
        confirm_user_link=AsyncMock(
            return_value=ConfirmUserLinkResponse(
                success=True, platform=platform, platform_user_id=platform_user_id
            )
        )
    )


@contextmanager
def _patched_slack_server_confirm(install):
    """Confirm a server link to Slack team T1, whose bot install is ``install``."""
    installs = MagicMock()
    installs.get_bot_install = AsyncMock(return_value=install)
    with (
        _patched_db(
            confirm_server_link=AsyncMock(
                return_value=ConfirmLinkResponse(
                    success=True,
                    platform="SLACK",
                    platform_server_id="T1",
                    server_name="eek",
                )
            )
        ),
        patch(f"{_R}.bot_installs_db", return_value=installs),
    ):
        yield


def _discord_meta() -> PlatformMeta:
    return PlatformMeta(
        platform="DISCORD",
        display_name="Discord",
        icon="discord.png",
        server_noun="server",
        enabled=True,
        add_bot_url="https://invite",
    )


def _slack_meta() -> PlatformMeta:
    return PlatformMeta(
        platform="SLACK",
        display_name="Slack",
        icon="slack.png",
        server_noun="workspace",
        enabled=True,
        add_bot_url="https://b.example/api/copilot-webhooks/slack/install",
    )
