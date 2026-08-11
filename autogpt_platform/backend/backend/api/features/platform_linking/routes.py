"""User-facing platform_linking REST routes (JWT auth)."""

import logging
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Path, Security
from pydantic import BaseModel, Field

from backend.copilot.bot.adapters.slack import oauth as slack_oauth
from backend.copilot.bot.adapters.slack import pending as slack_pending
from backend.copilot.bot.adapters.telegram import config as telegram_config
from backend.copilot.bot.adapters.telegram.login import verify_login
from backend.data.bot_installs import get_bot_install
from backend.data.db_accessors import platform_linking_db
from backend.platform_linking.models import (
    BotPlatformInfo,
    ConfirmLinkResponse,
    ConfirmUserLinkResponse,
    DeleteLinkResponse,
    LinkTokenInfoResponse,
    PendingInstallInfo,
    Platform,
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

from . import registry

logger = logging.getLogger(__name__)

router = APIRouter()

TokenPath = Annotated[
    str,
    Path(max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]


class ConfirmLinkRequest(BaseModel):
    """Optional confirm payload. ``telegram_auth`` carries the signed identity
    Telegram appends when the user reached this page via a login_url button —
    when present it must verify, and the link token must belong to that same
    Telegram user."""

    telegram_auth: dict[str, str] | None = Field(default=None)


def _verified_platform_user(body: ConfirmLinkRequest | None) -> str | None:
    if body is None or not body.telegram_auth:
        return None
    verified = verify_login(body.telegram_auth, telegram_config.get_bot_token())
    if verified is None:
        raise HTTPException(
            status_code=403, detail="Telegram login data failed verification."
        )
    return verified


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, NotAuthorizedError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, LinkAlreadyExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, LinkTokenExpiredError):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, LinkFlowMismatchError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal error.")


@router.get(
    "/tokens/{token}/info",
    response_model=LinkTokenInfoResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Get display info for a link token",
)
async def get_link_token_info_route(token: TokenPath) -> LinkTokenInfoResponse:
    try:
        info = await platform_linking_db().get_link_token_info(token)
    except (NotFoundError, LinkTokenExpiredError) as exc:
        raise _translate(exc) from exc
    info.server_noun = registry.server_noun_for(info.platform)
    return info


@router.post(
    "/tokens/{token}/confirm",
    response_model=ConfirmLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a SERVER link token (user must be authenticated)",
)
async def confirm_link_token(
    token: TokenPath,
    user_id: Annotated[str, Security(auth.get_user_id)],
    body: ConfirmLinkRequest | None = None,
) -> ConfirmLinkResponse:
    try:
        response = await platform_linking_db().confirm_server_link(
            token, user_id, verified_platform_user_id=_verified_platform_user(body)
        )
    except (
        NotFoundError,
        NotAuthorizedError,
        LinkFlowMismatchError,
        LinkTokenExpiredError,
        LinkAlreadyExistsError,
    ) as exc:
        raise _translate(exc) from exc
    response.return_url = await _slack_return_url_for_team(
        response.platform, response.platform_server_id
    )
    return response


@router.post(
    "/user-tokens/{token}/confirm",
    response_model=ConfirmUserLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a USER link token (user must be authenticated)",
)
async def confirm_user_link_token(
    token: TokenPath,
    user_id: Annotated[str, Security(auth.get_user_id)],
    body: ConfirmLinkRequest | None = None,
) -> ConfirmUserLinkResponse:
    try:
        response = await platform_linking_db().confirm_user_link(
            token, user_id, verified_platform_user_id=_verified_platform_user(body)
        )
    except (
        NotFoundError,
        NotAuthorizedError,
        LinkFlowMismatchError,
        LinkTokenExpiredError,
        LinkAlreadyExistsError,
    ) as exc:
        raise _translate(exc) from exc
    # The DM link completes the closed-loop journey: hand back a deep link to
    # the bot chat so the success page can return the user where they came
    # from. Slack's comes from the pending-install marker (retired here);
    # Telegram's is just the bot's public t.me address.
    if response.platform == Platform.SLACK.value:
        marker = await slack_pending.get_pending(user_id)
        if marker and marker.app_id:
            response.return_url = slack_pending.bot_dm_url(
                marker.app_id, marker.team_id
            )
        await slack_pending.clear_pending(user_id)
    elif response.platform == Platform.TELEGRAM.value:
        username = telegram_config.get_bot_username().lstrip("@")
        if username:
            response.return_url = f"https://t.me/{username}"
    return response


@router.get(
    "/links",
    response_model=list[PlatformLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all platform servers linked to the authenticated user",
)
async def list_my_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformLinkInfo]:
    return await platform_linking_db().list_server_links(user_id)


@router.get(
    "/user-links",
    response_model=list[PlatformUserLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all DM links for the authenticated user",
)
async def list_my_user_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformUserLinkInfo]:
    return await platform_linking_db().list_user_links(user_id)


@router.get(
    "/platforms",
    response_model=list[BotPlatformInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List bot platforms enabled on this deployment plus the caller's links",
    operation_id="list_bot_platforms",
)
async def list_bot_platforms(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[BotPlatformInfo]:
    db = platform_linking_db()
    user_links = await db.list_user_links(user_id)
    server_links = await db.list_server_links(user_id)
    dm_by_platform: dict[str, PlatformUserLinkInfo] = {}
    for link in user_links:
        dm_by_platform.setdefault(link.platform, link)
    servers_by_platform: dict[str, list[PlatformLinkInfo]] = {}
    for link in server_links:
        servers_by_platform.setdefault(link.platform, []).append(link)
    return [
        BotPlatformInfo(
            platform=meta.platform,
            display_name=meta.display_name,
            icon=meta.icon,
            server_noun=meta.server_noun,
            add_bot_url=_user_bound_install_url(
                meta.platform, meta.add_bot_url, user_id
            ),
            dm_link=dm_by_platform.get(meta.platform),
            server_links=servers_by_platform.get(meta.platform, []),
            pending_install=await _pending_install(
                meta.platform, user_id, dm_by_platform.get(meta.platform)
            ),
        )
        for meta in registry.enabled_platforms()
    ]


async def _slack_return_url_for_team(platform: str, team_id: str) -> str | None:
    """Deep link into the bot's DM for a just-linked Slack workspace."""
    if platform != Platform.SLACK.value:
        return None
    install = await get_bot_install(Platform.SLACK, team_id)
    if install is None or not install.app_id:
        return None
    return slack_pending.bot_dm_url(install.app_id, team_id)


def _user_bound_install_url(
    platform: str, add_bot_url: str | None, user_id: str
) -> str | None:
    """Attach the signed user param so the install can be attributed back.

    Only Slack's install flow round-trips our backend; other platforms'
    invite URLs go straight to the platform and are returned untouched.
    """
    if platform != Platform.SLACK.value or not add_bot_url:
        return add_bot_url
    return f"{add_bot_url}?u={slack_oauth.make_install_user_param(user_id)}"


async def _pending_install(
    platform: str, user_id: str, dm_link: PlatformUserLinkInfo | None
) -> PendingInstallInfo | None:
    """The caller's not-yet-linked install, if any. Linking the DM completes
    the journey, so an existing DM link clears the pending state."""
    if platform != Platform.SLACK.value or dm_link is not None:
        return None
    marker = await slack_pending.get_pending(user_id)
    if marker is None or not marker.app_id:
        return None
    return PendingInstallInfo(
        server_name=marker.team_name,
        open_bot_url=slack_pending.bot_dm_url(marker.app_id, marker.team_id),
    )


@router.delete(
    "/links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a platform server",
)
async def delete_link(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    try:
        return await platform_linking_db().delete_server_link(link_id, user_id)
    except (NotFoundError, NotAuthorizedError) as exc:
        raise _translate(exc) from exc


@router.delete(
    "/user-links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a DM / user link",
)
async def delete_user_link_route(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    try:
        return await platform_linking_db().delete_user_link(link_id, user_id)
    except (NotFoundError, NotAuthorizedError) as exc:
        raise _translate(exc) from exc
