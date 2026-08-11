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
from .pending import PendingSlackInstall, bot_dm_url, mark_pending

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

# ``u`` params are minted when the Bots page renders, and the page may sit
# open a while before the user clicks — allow much longer than the OAuth
# round-trip itself.
_USER_PARAM_TTL_SECONDS = 24 * 3600


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


async def _handle_install(request: Request) -> Response:
    # ``u`` is an HMAC-signed platform user id minted by the authenticated
    # Bots settings route. Folding it into the OAuth state is what lets the
    # callback attribute the install to an account; the public marketing-page
    # button has no ``u`` and installs anonymously.
    user_id = _verify_user_param(request.query_params.get("u", ""))
    params = urlencode(
        {
            "client_id": config.get_client_id(),
            "scope": ",".join(_SCOPES),
            "redirect_uri": _redirect_uri(),
            "state": _make_state(user_id),
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
    user_id = _verify_state(state)
    if user_id is None:  # "" is a valid anonymous install; None is a bad state.
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

    app_id = resp.get("app_id") or ""
    if user_id:
        await mark_pending(
            user_id,
            PendingSlackInstall(
                team_id=team_id, team_name=team.get("name"), app_id=app_id or None
            ),
        )
    if app_id:
        # Hand off to our own page rather than Slack's redirect page: it opens
        # the bot DM via the desktop deep link and then returns the tab to the
        # Bots settings page, so the install doesn't strand a dead browser tab.
        return RedirectResponse(
            _installed_page_url(
                team_id=team_id, app_id=app_id, bot_user_id=resp.get("bot_user_id")
            )
            or bot_dm_url(app_id, team_id),
            status_code=302,
        )
    return _done(ok=True, detail=team.get("name") or team_id)


def _installed_page_url(
    *, team_id: str, app_id: str, bot_user_id: str | None
) -> str | None:
    """Our post-install handoff page, or None when no frontend is configured."""
    base = (Settings().config.frontend_base_url or "").rstrip("/")
    if not base:
        return None
    params = {"team": team_id, "app": app_id}
    if bot_user_id:
        params["bot"] = bot_user_id
    return f"{base}/link/slack/installed?{urlencode(params)}"


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


def make_install_user_param(user_id: str) -> str:
    """Signed ``u`` value the authenticated Bots route appends to the install
    URL, so the unauthenticated OAuth flow can attribute the install to an
    account without trusting the query string."""
    payload = f"{user_id}.{int(time.time())}"
    return f"{payload}.{_sign(payload)}"


def _verify_user_param(value: str) -> str:
    """Return the user id from a valid ``u`` param, else "" (anonymous)."""
    try:
        user_id, ts, sig = value.rsplit(".", 2)
    except ValueError:
        return ""
    payload = f"{user_id}.{ts}"
    if not hmac.compare_digest(sig, _sign(payload)):
        return ""
    try:
        if (int(time.time()) - int(ts)) > _USER_PARAM_TTL_SECONDS:
            return ""
    except ValueError:
        return ""
    return user_id


def _make_state(user_id: str = "") -> str:
    nonce = secrets.token_urlsafe(24)
    payload = f"{nonce}.{int(time.time())}.{user_id}"
    return f"{payload}.{_sign(payload)}"


def _verify_state(state: str) -> str | None:
    """Return the embedded user id ("" for anonymous installs), or None when
    the state is forged or expired."""
    try:
        payload, sig = state.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload)):
        return None
    parts = payload.split(".")
    if len(parts) != 3:
        return None
    _nonce, ts, user_id = parts
    try:
        if (int(time.time()) - int(ts)) > _STATE_TTL_SECONDS:
            return None
    except ValueError:
        return None
    return user_id


def _sign(payload: str) -> str:
    key = config.get_client_secret().encode()
    return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
