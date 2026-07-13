"""Slack OAuth v2 install flow — the multi-workspace "Add to Slack" path.

    GET /install         → CSRF-signed state, 302 to Slack's authorize screen.
    GET /oauth/callback  → validate state, exchange the code for that workspace's
                           bot token (oauth.v2.access), store it encrypted per team.

Enabled only when the app's client id + secret are configured. The state is a
self-contained HMAC-signed nonce (keyed on the client secret) with a short TTL,
so the public, un-authenticated install button needs no server-side session.
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Callable, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from slack_sdk.web.async_client import AsyncWebClient

from backend.data.bot_analytics import record_guild_joined
from backend.data.bot_installs import upsert_bot_install
from backend.platform_linking.models import BotGuildInput, Platform
from backend.util.settings import Settings

from . import config

logger = logging.getLogger(__name__)

INSTALL_PATH = "/api/copilot-webhooks/slack/install"
CALLBACK_PATH = "/api/copilot-webhooks/slack/oauth/callback"

# Bot scopes requested at install — must stay in sync with app-manifest.yaml.
_SCOPES = (
    "app_mentions:read",
    "channels:history",
    "channels:read",
    "chat:write",
    "chat:write.public",
    "commands",
    "files:read",
    "files:write",
    "im:history",
    "im:read",
    "im:write",
    "users:read",
    "team:read",
)

_STATE_TTL_SECONDS = 600


def is_enabled() -> bool:
    """The install flow needs OAuth app credentials; single-workspace mode
    (static token only) runs without them."""
    return bool(config.get_client_id() and config.get_client_secret())


def register_routes(
    app: FastAPI, on_installed: Optional[Callable[[str], None]] = None
) -> None:
    # on_installed(team_id) fires after a successful (re)install so the adapter
    # can drop any cached client built from the workspace's previous token.
    if not is_enabled():
        return

    async def _callback(request: Request) -> Response:
        return await _handle_callback(request, on_installed)

    app.add_api_route(INSTALL_PATH, _handle_install, methods=["GET"])
    app.add_api_route(CALLBACK_PATH, _callback, methods=["GET"])


def _redirect_uri() -> str:
    base = Settings().config.platform_base_url.rstrip("/")
    return f"{base}{CALLBACK_PATH}"


async def _handle_install() -> Response:
    params = urlencode(
        {
            "client_id": config.get_client_id(),
            "scope": ",".join(_SCOPES),
            "redirect_uri": _redirect_uri(),
            "state": _make_state(),
        }
    )
    return RedirectResponse(
        f"https://slack.com/oauth/v2/authorize?{params}", status_code=302
    )


async def _handle_callback(
    request: Request, on_installed: Optional[Callable[[str], None]] = None
) -> Response:
    if error := request.query_params.get("error"):
        # User declined, or Slack rejected the request.
        return _done(ok=False, detail=error)
    state = request.query_params.get("state", "")
    code = request.query_params.get("code", "")
    if not _verify_state(state):
        return PlainTextResponse("invalid or expired state", status_code=400)
    if not code:
        return PlainTextResponse("missing code", status_code=400)

    resp = await AsyncWebClient().oauth_v2_access(
        client_id=config.get_client_id(),
        client_secret=config.get_client_secret(),
        code=code,
        redirect_uri=_redirect_uri(),
    )
    if not resp.get("ok"):
        logger.warning("Slack oauth.v2.access failed: %s", resp.get("error"))
        return _done(ok=False, detail=resp.get("error") or "exchange failed")

    token = resp.get("access_token") or ""
    team = resp.get("team") or {}
    team_id = team.get("id") or ""
    if not token or not team_id:
        return _done(ok=False, detail="incomplete install response")

    await upsert_bot_install(
        platform=Platform.SLACK,
        team_id=team_id,
        bot_token=token,
        bot_user_id=resp.get("bot_user_id"),
        app_id=resp.get("app_id"),
        name=team.get("name"),
    )
    if on_installed is not None:
        on_installed(team_id)
    try:
        await record_guild_joined(
            BotGuildInput(
                platform=Platform.SLACK, server_id=team_id, name=team.get("name")
            )
        )
    except Exception:
        logger.warning(
            "Failed to record BotGuild for Slack install %s", team_id, exc_info=True
        )
    return _done(ok=True, detail=team.get("name") or team_id)


def _done(*, ok: bool, detail: str) -> Response:
    """Land the browser back on the settings page (or a plain confirmation if no
    frontend URL is configured)."""
    base = (Settings().config.frontend_base_url or "").rstrip("/")
    if base:
        flag = "slack_installed" if ok else "slack_install_error"
        return RedirectResponse(f"{base}/settings/bots?{flag}=1", status_code=302)
    if ok:
        return PlainTextResponse(
            f"AutoGPT was added to {detail}. You can close this tab and run "
            "/setup in Slack to link your account."
        )
    return PlainTextResponse(f"Slack install failed: {detail}", status_code=400)


def _make_state() -> str:
    nonce = secrets.token_urlsafe(24)
    payload = f"{nonce}.{int(time.time())}"
    return f"{payload}.{_sign(payload)}"


def _verify_state(state: str) -> bool:
    try:
        nonce, ts, sig = state.rsplit(".", 2)
    except ValueError:
        return False
    payload = f"{nonce}.{ts}"
    if not hmac.compare_digest(sig, _sign(payload)):
        return False
    try:
        return (int(time.time()) - int(ts)) <= _STATE_TTL_SECONDS
    except ValueError:
        return False


def _sign(payload: str) -> str:
    key = config.get_client_secret().encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
